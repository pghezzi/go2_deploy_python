"""Shadow integration tests: real classifiers, synthetic sensors, no transport."""
import ast
import copy
import json
import io
import struct
import subprocess
import sys
from unittest.mock import patch, Mock
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch

from shadow_experiment.core import (GroundTruth, StateBuffer, Pipelines, TrialWriter,
                                    load_config, preprocess, read_trial)
from shadow_experiment.collect import Collector
from shadow_experiment.analyze import replay, report, trial_metrics, validate
from shadow_experiment.benchmark import benchmark
from terrain_selector import TerrainSelector

ROOT = Path(__file__).resolve().parents[1]


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.c = load_config(ROOT/'configs/shadow/rough_to_gap.yaml')
        self.c['output_root'] = self.tmp.name
        self.c['runtime'].update(update_hz=1000, max_input_age_s=10, chunk_frames=2)
        self.c['analysis']['max_contiguous_gap_s'] = 10

    def test_gamepad_b_is_edge_triggered_and_latches_once(self):
        from types import SimpleNamespace
        from common.remote_controller import KeyMap
        c = Collector(self.c)
        def state(tick, *keys):
            packet = bytearray(40)
            struct.pack_into('H', packet, 2, sum(1 << k for k in keys))
            c.state(SimpleNamespace(tick=tick, wireless_remote=packet,
                                   imu_state=SimpleNamespace(rpy=[0.,0.,0.], gyroscope=[0.,0.,0.])))
        try:
            state(1, KeyMap.B)  # Already held at startup: no annotation.
            state(2, KeyMap.B)
            self.assertIsNone(c.truth.event)
            state(3)
            state(4, KeyMap.B, KeyMap.Y)  # Avoid simultaneous control shortcuts.
            self.assertIsNone(c.truth.event)
            state(5)
            state(6, KeyMap.B)
            event = dict(c.truth.event)
            self.assertEqual(event['input_source'], 'gamepad')
            self.assertEqual(event['buttons'], ['B'])
            self.assertEqual(event['sensor_tick'], 6)
            self.assertEqual(event['timestamp_ns'], c.states.samples[-1]['receipt_ns'])
            self.assertIsNone(c.truth.crossing)
            state(7, KeyMap.B)
            state(8)
            state(9, KeyMap.B)
            self.assertEqual(c.truth.event, event)
            self.assertIsNone(event['first_frame_id'])
        finally:
            c.writer.close('test', c.truth, dict(c.counters))
        events = [json.loads(line) for line in (c.writer.path/'events.jsonl').read_text().splitlines()]
        self.assertEqual(sum(e['kind']=='transition' for e in events), 1)
        self.assertEqual(sum(e['kind']=='gamepad_marker' for e in events), 2)

    def test_gamepad_disabled_and_duplicate_state_packets(self):
        from types import SimpleNamespace
        c = Collector(self.c)
        packet = bytearray(40)
        state = SimpleNamespace(tick=1, wireless_remote=packet,
                                imu_state=SimpleNamespace(rpy=[0.,0.,0.], gyroscope=[0.,0.,0.]))
        try:
            c.state(state)
            struct.pack_into('H', packet, 2, 1 << 9)
            c.state(state)  # Same sensor tick cannot create a new press.
            self.assertIsNone(c.truth.event)
            c.config['transition']['gamepad']['enabled'] = False
            state.tick = 2
            c.state(state)
            self.assertIsNone(c.truth.event)
        finally:
            c.writer.close('test', c.truth, dict(c.counters))

    def test_transition_latches_all_trigger_types(self):
        for trigger in ({'type':'time','seconds':1}, {'type':'operator','marker':'go'},
                        {'type':'position','axis':'x','comparison':'ge','threshold_m':2}):
            cfg = dict(self.c, transition=trigger)
            truth = GroundTruth(cfg, 100)
            self.assertEqual(truth.label(100), 'rough')
            truth.observe(100, position={'xyz':[0,0,0],'receipt_ns':100}, marker='wrong')
            self.assertIsNone(truth.event)
            truth.observe(2000000100, frame_id=3, position={'xyz':[3,0,0],'receipt_ns':2000000100}, marker='go')
            event = dict(truth.event)
            truth.observe(3000000100, position={'xyz':[0,0,0],'receipt_ns':3000000100})
            self.assertEqual(truth.event, event)
            self.assertEqual(truth.label(3000000100), 'gap')
            self.assertIsNone(truth.crossing)
            self.assertEqual(GroundTruth(cfg,100).label(100), 'rough')

    def test_causal_alignment_never_uses_future_or_stale_state(self):
        b = StateBuffer(2)
        b.add({'receipt_ns':100})
        b.add({'receipt_ns':300})
        self.assertEqual(b.causal(200,1)['receipt_ns'],100)
        self.assertIsNone(b.causal(50,1))
        self.assertIsNone(b.causal(200,1e-9))
        b.add({'receipt_ns':400})
        self.assertEqual(b.evictions,1)

    def test_training_preprocessing_matches_reference_operations(self):
        raw = np.random.default_rng(0).integers(0,4000,(480,640),dtype=np.uint16)
        actual = preprocess(raw,.001,self.c['camera'])
        depth = torch.tensor(raw.astype(np.float32)) * .001
        depth = depth.clamp(0,3)/3
        expected = torch.nn.functional.interpolate(depth[48:,:][...,28:-36][None,None],size=(48,64),mode='bicubic',align_corners=False)[0,0].clamp(0,1).numpy()
        np.testing.assert_array_equal(actual,expected)

    def test_shared_filters_match_existing_update_and_reset(self):
        bank = Pipelines(self.c)
        depth = np.random.default_rng(2).uniform(0,1,(48,64)).astype('f')
        for name,spec in self.c['models'].items():
            for mode in ('instantaneous','ema','bayes'):
                selector=TerrainSelector(spec['path'],label_to_lora={k:i for i,k in enumerate(self.c['class_mapping'])},mode=mode,**self.c['filters'])
                bank.reset()
                for _ in range(3):
                    result=bank.run(depth,[.02,-.1,0],[.1,0,.2])['selectors'][name+'/'+mode]
                    original=selector.update(depth,[.02,-.1,0],[.1,0,.2])
                    self.assertEqual(result['proposed_skill'],self.c['class_mapping'][original['label']])
                    if mode=='bayes':
                        np.testing.assert_array_equal(result['distribution'],selector.belief.tolist())
                bank.reset()
                self.assertTrue(all(s.selected_index is None for s in bank.filters.values()))

    def make_trial(self, finish=True, transition=True):
        c=Collector(self.c)
        raw=np.random.default_rng(5).integers(0,3000,(480,640),dtype=np.uint16)
        for i in range(6):
            stamp=time.monotonic_ns()
            if i==3 and transition:
                event=c.truth.observe(stamp,marker='transition')
                c.writer.event('transition',**event)
            c.states.add({'receipt_ns':stamp-1000000,'sensor_tick':i,'rpy':[0.,.1,0.], 'omega':[0.,0.,0.]})
            c.process({'frame_id':i,'receipt_ns':stamp,'sensor_timestamp_ms':float(i*100),
                       'sensor_clock':'synthetic', 'depth_scale_m':.001,'rgb_timestamp_ms':None},raw,None)
        if finish:
            c.writer.close('test_fixture',c.truth,dict(c.counters))
        return c

    def test_end_to_end_replay_report_and_isolated_benchmark(self):
        c=self.make_trial()
        result=replay(c.writer.path)
        self.assertTrue(result['consistent'],result)
        manifest,rows,issues=read_trial(c.writer.path)
        self.assertEqual(len(rows),6)
        issues,event=validate(manifest,rows,issues)
        self.assertEqual(issues,[])
        report(self.tmp.name,Path(self.tmp.name)/'report',appendix=True)
        self.assertTrue((Path(self.tmp.name)/'report/aggregate.csv').exists())
        self.assertTrue(list((Path(self.tmp.name)/'report').glob('*.pdf')))
        benchmark(c.writer.path,Path(self.tmp.name)/'bench',repeats=1,warmup=1)
        self.assertTrue((Path(self.tmp.name)/'bench/isolated_summary.csv').exists())
        with self.assertRaises(FileExistsError):
            TrialWriter(self.c,c.pipelines.metadata,time.monotonic_ns())

    def test_interruption_recovers_atomic_unindexed_chunks(self):
        c=self.make_trial(finish=False)
        c.writer.events.close()
        path=c.writer.path/'manifest.json'
        manifest=json.loads(path.read_text()); manifest['chunks']=manifest['chunks'][:-1]; manifest['frame_count']=sum(c['frames'] for c in manifest['chunks'])
        path.write_text(json.dumps(manifest))
        (c.writer.path/'frames_000003.tmp').write_bytes(b'incomplete')
        manifest,rows,issues=read_trial(c.writer.path)
        self.assertEqual(len(rows),6)
        self.assertIn('unclosed_trial',issues)
        self.assertTrue(any(i.startswith('recovered_unindexed_chunk') for i in issues))
        self.assertIn('unfinished_temporary_write',issues)

    def test_duplicate_stale_and_queue_overflow_do_not_advance_filters(self):
        c=self.make_trial()
        before=c.counters['accepted']
        c.writer.events=(c.writer.path/'events.jsonl').open('a')
        raw=np.zeros((480,640),np.uint16)
        c.process({'frame_id':5,'receipt_ns':time.monotonic_ns(),'sensor_timestamp_ms':500.},raw,None)
        self.assertEqual(c.counters['accepted'],before)
        for i in range(5):
            c.offer({},raw,None)
        self.assertEqual(c.counters['capture_queue_drop_new'],3)
        c.writer.events.close()

    def test_segment_bounded_delay_cannot_match_after_invalid_interval(self):
        c=self.make_trial()
        manifest,rows,_=read_trial(c.writer.path)
        for i,(r,_,_) in enumerate(rows):
            for result in r['selectors'].values():
                result['proposed_skill']='gap' if i==5 else 'rough'
            if i==4:
                r['valid_for_analysis']=False
        results=trial_metrics(manifest,rows,manifest['transition'],c.writer.path)
        self.assertTrue(all(r['transition_missed'] for r in results))
        self.assertTrue(all(r['first_match_delay_s'] is None for r in results))

    def test_uncertain_interval_between_frames_breaks_transition_segment(self):
        from shadow_experiment.analyze import segments
        rows = [({'elapsed_s': t, 'valid_for_analysis': True}, {}, None) for t in (0., .2)]
        valid, ids = segments(rows, 1., [(0.05, .15)])
        self.assertEqual(valid, [0,1])
        self.assertNotEqual(ids[0], ids[1])

    def test_independent_trials_reset_and_approach_only_is_not_a_transition(self):
        first = self.make_trial(transition=False)
        self.c['trial_id'] = '004'
        second = self.make_trial(transition=False)
        a, rows_a, _ = read_trial(first.writer.path)
        b, rows_b, _ = read_trial(second.writer.path)
        self.assertTrue(a['approach_only'] and b['approach_only'])
        self.assertEqual(rows_b[0][0]['accepted_index'], 0)
        for key in rows_a[0][0]['selectors']:
            np.testing.assert_array_equal(rows_a[0][0]['selectors'][key]['distribution'],
                                          rows_b[0][0]['selectors'][key]['distribution'])
        metrics = trial_metrics(b, rows_b, None, second.writer.path)
        self.assertTrue(all(m['approach_only'] and not m['transition_eligible'] for m in metrics))

    def test_hard_process_exit_retains_only_committed_chunks(self):
        script = """
import os, time
import numpy as np
from shadow_experiment.core import TrialWriter, load_config
c = load_config('configs/shadow/rough_to_gap.yaml')
c['output_root'] = os.environ['SHADOW_TEST_OUTPUT']
c['runtime']['chunk_frames'] = 2
w = TrialWriter(c, {}, time.monotonic_ns())
for i in range(3):
    w.add({'frame_id': i}, np.zeros((2,2),np.uint16), np.zeros((48,64),np.float32), [0,0,0], [0,0,0])
os._exit(9)
"""
        import os
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=str(ROOT),
                                env=dict(os.environ, SHADOW_TEST_OUTPUT=self.tmp.name), timeout=30)
        self.assertEqual(result.returncode, 9)
        path = next(Path(self.tmp.name).rglob('manifest.json')).parent
        manifest, rows, issues = read_trial(path)
        self.assertEqual(len(rows), 2)
        self.assertFalse(manifest['complete'])
        self.assertIn('unclosed_trial', issues)

    def test_annotation_and_verified_crossing_are_separate(self):
        truth = GroundTruth(self.c, 0)
        truth.verify_crossing(100, 'observer saw feet cross')
        self.assertEqual(truth.label(200), 'rough')
        self.assertIsNone(truth.event)
        truth.observe(300, marker='transition')
        self.assertEqual(truth.label(200), 'rough')
        self.assertEqual(truth.label(300), 'gap')

    def test_fragmented_camera_transport_preserves_raw_input(self):
        from shadow_experiment.camera_tap import recv_packet
        raw = np.arange(12, dtype=np.uint16).reshape(3,4)
        data=io.BytesIO()
        np.savez(data,raw=raw,metadata=np.frombuffer(b'{"frame_id": 7}',np.uint8))
        wire=io.BytesIO(struct.pack('!I',len(data.getvalue()))+data.getvalue())
        conn=Mock()
        conn.recv.side_effect=lambda n: wire.read(min(n,7))
        meta, received, rgb=recv_packet(conn)
        self.assertEqual(meta['frame_id'],7)
        np.testing.assert_array_equal(raw,received)
        self.assertIsNone(rgb)
        with self.assertRaises(EOFError):
            recv_packet(conn)

    def test_main_only_initializes_state_subscribers(self):
        from shadow_experiment import collect
        from unitree_sdk2py.core import channel
        fake=Mock()
        with patch.object(channel,'ChannelPublisher',side_effect=AssertionError('publication forbidden')) as publisher, \
             patch.object(channel,'ChannelFactoryInitialize'), patch.object(channel,'ChannelSubscriber') as subscriber, \
             patch.object(collect,'Collector',return_value=fake), patch.object(collect.signal,'signal'), \
             patch.object(sys,'argv',['collect','--config',str(ROOT/'configs/shadow/rough_to_gap.yaml')]):
            collect.main()
        publisher.assert_not_called()
        self.assertEqual([c.args[0] for c in subscriber.call_args_list],['rt/lowstate','rt/sportmodestate'])
        fake.run.assert_called_once()

    def test_vendored_metrics_equal_inspected_reference(self):
        from shadow_experiment import reference_metrics
        source=Path(self.c['reference_repo'])/'legged_gym/utils/depth_terrain_classifier/terrain_classifier_bayes_streaming_prototype_rbf.py'
        if not source.exists():
            self.skipTest('Reference checkout unavailable')
        original=ast.parse(source.read_text())
        copied=ast.parse(Path(reference_metrics.__file__).read_text())
        for name in ('evaluate_transition_accounting','_false_transition_rate'):
            a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in copied.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b))

    def test_shadow_collector_has_no_motor_publication_or_control_imports(self):
        forbidden={'ChannelPublisher','SportClient','MotionSwitcherClient','LowCmd_', 'DepthWaQController','SinglePolicyController'}
        for file in (ROOT/'shadow_experiment').glob('*.py'):
            tree=ast.parse(file.read_text())
            names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
            imports={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
            self.assertFalse(names & forbidden,(file,names & forbidden))
            self.assertNotIn('controller',imports)


if __name__=='__main__':
    unittest.main()

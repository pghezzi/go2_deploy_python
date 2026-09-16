import unittest
from unittest.mock import patch

from shadow_experiment.live_latency import sample_memory, summarize_samples


class MemoryTests(unittest.TestCase):
    def test_rss_uses_resident_pages_and_real_page_size(self):
        with patch('pathlib.Path.read_text', return_value='9999 512 0 0 0 0 0'), patch('os.sysconf', return_value=4096):
            result = sample_memory('cpu')
        self.assertEqual(result['process_rss_mib'], 2.)
        self.assertIsNone(result['cuda_allocated_mib'])
        self.assertIsNone(result['cuda_reserved_mib'])

    def test_cuda_allocations_are_separate_from_rss(self):
        with patch('pathlib.Path.read_text', return_value='9999 512'), patch('os.sysconf', return_value=4096), patch('torch.cuda.memory_allocated', return_value=3*2**20), patch('torch.cuda.memory_reserved', return_value=4*2**20):
            result = sample_memory('cuda:0')
        self.assertEqual(result, dict(process_rss_mib=2., cuda_allocated_mib=3., cuda_reserved_mib=4.))

    def test_mean_sample_std_and_unavailable_cuda(self):
        rows = [dict(total_ms=1., classifier_ms=.8, filter_ms=.2, process_rss_mib=v,
                     cuda_allocated_mib=None, cuda_reserved_mib=None) for v in (2.,4.,6.)]
        summary = summarize_samples('feature/ema', rows)
        self.assertEqual(summary['process_rss_mib_mean'], 4.)
        self.assertEqual(summary['process_rss_mib_std'], 2.)
        self.assertEqual(summary['total_ms_std'], 0.)
        self.assertIsNone(summary['cuda_allocated_mib_mean'])

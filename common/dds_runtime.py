"""Select the robot's compatible DDS runtime before importing native bindings."""
import os
from pathlib import Path
import platform
import sys


def configure_dds_runtime():
    # Never replace the system DDS used by Unitree services. Scope selection to
    # this process and its children. Re-exec is needed for ELF dependency lookup.
    explicit = os.environ.get('GO2_CYCLONEDDS_HOME')
    if explicit:
        root = Path(explicit).expanduser().resolve()
    elif platform.machine() == 'aarch64':
        root = Path.home() / 'cyclonedds-0.10.2'
        if not (root / 'lib/libddsc.so').is_file():
            return
    else:
        return
    if not (root / 'lib/libddsc.so').is_file():
        raise RuntimeError('Missing DDS library under ' + str(root))
    library = str(root / 'lib')
    paths = os.environ.get('LD_LIBRARY_PATH', '').split(':')
    if os.environ.get('CYCLONEDDS_HOME') == str(root) and paths[0] == library:
        return
    env = dict(os.environ, CYCLONEDDS_HOME=str(root),
               LD_LIBRARY_PATH=':'.join([library] + [p for p in paths if p and p != library]))
    main = sys.modules['__main__']
    spec = getattr(main, '__spec__', None)
    invocation = ['-m', spec.name] if spec is not None else [sys.argv[0]]
    print('DDS runtime: ' + str(root), flush=True)
    os.execve(sys.executable, [sys.executable, '-u'] + invocation + sys.argv[1:], env)

"""Run the selected competition model with the official input/output contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

VERSION = 'official_deployment_f1_policy_v3'


def selected_model(base):
    base = Path(base).resolve()
    selection = json.loads((base/'release/final_submission.json').read_text(encoding='utf-8'))
    expected = f'models/event_stack/{VERSION}/deployment'
    if selection.get('run_key') != VERSION or selection.get('bundle') != expected:
        raise ValueError('final model selection does not match the submitted version')
    bundle = base/expected
    from src.pipeline.artifacts import _is_link_or_reparse_point
    for path in (base/'models', bundle.parent.parent, bundle.parent, bundle):
        if _is_link_or_reparse_point(path): raise ValueError('linked model directories are not supported')
    actual = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir() if p.is_file()}
    if actual != selection.get('deployment_sha256'):
        raise ValueError('final model file set or SHA-256 differs')
    return selection, bundle


def main(argv=None, *, base=None):
    base = Path(base or Path(__file__).resolve().parents[1]).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--self-check',action='store_true')
    modes.add_argument('--visual',action='store_true')
    parser.add_argument('--port',type=int,default=None)
    parser.add_argument('--open',dest='open_browser',action='store_true')
    parser.add_argument('--no-open',dest='open_browser',action='store_false')
    parser.set_defaults(open_browser=False)
    args = parser.parse_args(argv)
    if args.port is not None and not 0 <= args.port <= 65535: parser.error('port must be between 0 and 65535')
    if not args.visual and (args.port is not None or args.open_browser): parser.error('port/browser options require --visual')
    try:
        selection,bundle = selected_model(base)
        if args.visual:
            from src.pipeline.inference import local_server
            from threadpoolctl import threadpool_limits
            arguments=['--bundle',str(bundle),'--run-key',selection['run_key'],'--visual-dir',str(base/'visual'),
                '--host','127.0.0.1','--port',str(4173 if args.port is None else args.port)]
            if args.open_browser: arguments.append('--open')
            with tempfile.TemporaryDirectory(prefix='eatingsense-final-',ignore_cleanup_errors=True) as work:
                with threadpool_limits(limits=min(8,os.cpu_count() or 1)):
                    return local_server.main(arguments,temp_root=Path(work))
        from src.pipeline.inference import Predictor
        from src.pipeline.inference.official_runner import run
        predictor=Predictor.from_bundle(bundle,run_key=selection['run_key'],device='cpu')
        if args.self_check:
            print(json.dumps(dict(status='OK',run_key=selection['run_key'],device='cpu')))
        else:
            print(json.dumps(run(base,bundle,predictor=predictor),ensure_ascii=True))
        return 0
    except (OSError,ValueError,RuntimeError,AssertionError,KeyError,TypeError) as exc:
        print(f'Competition inference failed: {exc}. No partial results.csv was published.',file=sys.stderr)
        return 2


if __name__=='__main__': raise SystemExit(main())

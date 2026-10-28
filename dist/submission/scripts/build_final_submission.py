"""围绕官方已评测的原始 EXE 组装最终源码包。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from scripts.build_competition_exe import sha, write_manifest, verify_submission, clean_environment
from scripts.submission_sources import add_sources, copy_file
from scripts.final_submission_entry import selected_model, VERSION
from scripts.build_inference_distribution import _atomic_replace

EXPECTED_EXE = 'fa6987b6244fe73c76ee741ea352fcc74e8e4de06de332812e99b1f1d963aa44'
FINAL_FILES = ('build_final_submission.py','final_submission_entry.py','final_submission_readme.md')


def final_executable(root):
    root=Path(root)
    source=root/'dist/official_trials/deployment_f1_policy_v3/EatingDetection.exe'
    if not source.is_file(): source=root/'EatingDetection.exe'
    if not source.is_file() or sha(source)!=EXPECTED_EXE:
        raise ValueError('the exact officially evaluated executable is required')
    return source


def copy_selected_bundle(source,target):
    """保留已校验模型文件的原始字节，包括 JSON 换行符。"""
    source,target=Path(source),Path(target)
    if target.exists(): raise FileExistsError('model copy destination already exists')
    if source.is_symlink() or any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('linked model files are not supported')
    shutil.copytree(source,target)


def install_notices(root,package):
    """重新组装时复用冻结程序附带的完整第三方声明。"""
    root,package=Path(root),Path(package)
    for source in (root/'THIRD_PARTY_NOTICES.txt',root/'dist/submission/THIRD_PARTY_NOTICES.txt'):
        if source.is_file():
            copy_file(source,package/'THIRD_PARTY_NOTICES.txt')
            return
    raise ValueError('complete shipped third-party notices are required for final reassembly')


def verify_final(package):
    package=Path(package)
    manifest=verify_submission(package)
    selection,_=selected_model(package)
    if (manifest['release_run_key']!=VERSION or sha(package/'EatingDetection.exe')!=EXPECTED_EXE
        or manifest.get('inference_model')!=VERSION or manifest.get('official_f1')!=.5333333333333333):
        raise ValueError('final package inference identity differs')
    if selection['executable_sha256']!=EXPECTED_EXE: raise ValueError('selected EXE SHA differs')
    from src.pipeline.artifacts import load_current_promoted_release
    if load_current_promoted_release(package)['run_key']!='160afaf81debf1ee':
        raise ValueError('historical reproduction evidence is missing')
    return manifest


def final_output(root):
    root=Path(root).resolve();target=root/'submission';output=target/'EatingDetection.exe'
    from src.pipeline.artifacts import _is_link_or_reparse_point
    if _is_link_or_reparse_point(target) or _is_link_or_reparse_point(output):
        raise ValueError('linked final output is forbidden')
    if target.exists():
        if not target.is_dir() or any(p.name!='EatingDetection.exe' for p in target.iterdir()):
            raise ValueError('root submission must contain only the final EXE')
        if output.exists() and (not output.is_file() or sha(output)!=EXPECTED_EXE):
            raise ValueError('unexpected existing root executable')
    return target,output


def publish_final(staging,dest,root):
    """先发布固定版本的根目录 EXE；源码包安装失败时撤销本次新增的 EXE。"""
    staging,dest=Path(staging),Path(dest);target,output=final_output(root)
    created=False;temporary=None
    try:
        if not output.exists():
            target.mkdir(exist_ok=True)
            handle,filename=tempfile.mkstemp(prefix='.EatingDetection.',suffix='.tmp',dir=target)
            os.close(handle);temporary=Path(filename)
            shutil.copyfile(staging/'EatingDetection.exe',temporary)
            if sha(temporary)!=EXPECTED_EXE: raise ValueError('staged root executable SHA differs')
            if output.exists(): raise FileExistsError('root executable appeared during publication')
            os.replace(temporary,output);temporary=None;created=True
        _atomic_replace(staging,dest)
        return dest
    except BaseException:
        # 已有且正确的 EXE 保持不变，重新组装时不覆盖。
        completed=(not staging.exists() and (dest/'EatingDetection.exe').is_file()
            and sha(dest/'EatingDetection.exe')==EXPECTED_EXE)
        if created and not completed and output.is_file() and sha(output)==EXPECTED_EXE:
            output.unlink()
        raise
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def build(root=ROOT):
    root=Path(root).resolve();selection,bundle=selected_model(root);exe=final_executable(root)
    historical=selection['reproduction_run_key'];dest=root/'dist/submission'
    final_output(root)
    for path in (root/'dist',dest):
        if path.is_symlink(): raise ValueError('linked destination forbidden')
    if (dest/'ACC_test_data').exists() or (dest/'results.csv').exists():
        raise ValueError('runtime data/results must be preserved; move them before rebuilding')
    dest.parent.mkdir(exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.submission.staging-',dir=dest.parent))
    try:
        shutil.copyfile(exe,staging/'EatingDetection.exe')
        add_sources(root,staging,historical)
        install_notices(root,staging)
        copy_selected_bundle(bundle,staging/selection['bundle'])
        for name in FINAL_FILES: copy_file(root/'scripts'/name,staging/'scripts'/name)
        copy_file(root/'release/final_submission.json',staging/'release/final_submission.json')
        copy_file(root/'scripts/final_submission_readme.md',staging/'README.md')
        # 源码和可视化入口选择与 EXE 相同的模型。
        (staging/'main.py').write_text('"""竞赛正式推理入口。"""\nimport sys\n'
            'sys.dont_write_bytecode = True\nfrom pathlib import Path\n'
            'sys.path.insert(0,str(Path(__file__).resolve().parent))\n'
            'from scripts.final_submission_entry import main\n'
            'if __name__ == "__main__": raise SystemExit(main())\n',encoding='utf-8',newline='\n')
        (staging/'serve.py').write_text('"""本地可视化服务源码入口。"""\nimport sys\n'
            'sys.dont_write_bytecode = True\nfrom pathlib import Path\n'
            'sys.path.insert(0,str(Path(__file__).resolve().parent))\n'
            'from scripts.final_submission_entry import main\n'
            'if __name__ == "__main__": raise SystemExit(main(["--visual",*sys.argv[1:]]))\n',encoding='utf-8',newline='\n')
        api=root/'src/pipeline/inference/official_api'
        for name in ('input.py','output.py','provenance.json'): copy_file(api/name,staging/'official_api'/name)
        source_files={p.relative_to(staging).as_posix():sha(p) for tree in ('src','scripts') for p in (staging/tree).rglob('*.py')}
        original=json.loads((root/'outputs/official_recovery/deployment_f1_policy_v3_exe/summary.json').read_text()) if (root/'outputs/official_recovery/deployment_f1_policy_v3_exe/summary.json').is_file() else {}
        metadata=dict(format='official-20261015-windows-x64-onefile',release_run_key=VERSION,
            inference_model=VERSION,official_f1=selection['official_metrics']['f1'],official_metrics=selection['official_metrics'],
            historical_reproduction_run_key=historical,historical_development_f1=.6514285714285715,
            source_delivery=True,visual_delivery=True,executable_reused=True,exe_sha256=EXPECTED_EXE,
            deployment_sha256=selection['deployment_sha256'],source_sha256=source_files,
            executable_code_commit=original.get('code_sha','4d4418364e1d6ec742e80fe47fa3517aecea7e8c'),
            official_api=json.loads((api/'provenance.json').read_text(encoding='utf-8')),
            score_scope='official feedback distinct from historical local cross-validation',
            no_new_training=True)
        write_manifest(staging,metadata);verify_final(staging)
        # 替换旧包前检查源码与程序的模型选择，不一致时停止发布。
        for command in ([str(staging/'EatingDetection.exe'),'--self-check'],[sys.executable,'-B',str(staging/'main.py'),'--self-check']):
            result=subprocess.run(command,cwd=root,env=clean_environment(),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
            if result.returncode or json.loads(result.stdout.strip().splitlines()[-1]).get('run_key')!=VERSION:
                raise RuntimeError('final EXE/source self-check failed')
        return publish_final(staging,dest,root)
    finally:
        if staging.exists():
            if staging.parent!=dest.parent or not staging.name.startswith('.submission.staging-'):
                raise RuntimeError('unexpected staging cleanup target')
            shutil.rmtree(staging)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:
        package=ROOT if (ROOT/'build_manifest.json').is_file() else ROOT/'dist/submission'
        print(json.dumps(verify_final(package),ensure_ascii=False))
    else: print(build())

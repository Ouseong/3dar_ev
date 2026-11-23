"""Build the code-only handoff ZIP. Includes no recordings, features or weights."""
from pathlib import Path
import hashlib,json,zipfile


def main():
    root=Path(__file__).resolve().parents[1]
    output=root.parent/'3DAR_EV_Claude_Handoff_20261115.zip'
    top_files={'README.md','CLAUDE.md','HANDOFF_FOR_CLAUDE.md','FOR_CLAUDE_PROMPT.txt',
        'pyproject.toml','requirements.txt','run_spark.sh','smoke_spark.sh','.gitignore'}
    allowed_dirs={'ev3dar','configs','scripts','tests','prepared','validation','references'}
    files=[]
    for p in root.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(root)
        if '__pycache__' in rel.parts or p.suffix in {'.pyc','.wav','.mp3','.flac','.npz','.pt','.pth','.zip'}:continue
        if len(rel.parts)==1:
            if rel.name not in top_files:continue
        elif rel.parts[0] not in allowed_dirs:continue
        if p.suffix not in {'.py','.json','.jsonl','.ipynb','.md','.txt','.toml','.sh','.gitignore'} and p.name!='.gitignore':
            raise ValueError('Unexpected package extension: '+str(rel))
        files.append(p)
    files.sort(key=lambda p:p.relative_to(root).as_posix())
    manifest=dict(package='EV_3DAR_AI',date='2026-11-15',measurement_data_included=False,
        trained_weights_included=False,files=[dict(path=p.relative_to(root).as_posix(),
            bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    mp=root/'PACKAGE_MANIFEST.json'
    mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    files.append(mp)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in files:archive.write(p,'EV_3DAR_AI/'+p.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert not any(n.lower().endswith(('.wav','.npz','.pt','.pth')) for n in archive.namelist())
        for item in manifest['files']:
            assert hashlib.sha256(archive.read('EV_3DAR_AI/'+item['path'])).hexdigest()==item['sha256']
    print(json.dumps(dict(zip=str(output),files=len(files),bytes=output.stat().st_size,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),crc_and_member_hashes_ok=True),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

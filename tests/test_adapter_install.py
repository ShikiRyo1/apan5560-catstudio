"""Installer checks with synthetic byte fixtures, not model-inference tests."""
import hashlib, json, sys, tempfile, unittest, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fetch_adapter import ALLOWED, install_archive

class AdapterInstallTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.root = Path(self.workspace.name)
        self.archive = self.root/'fixture.zip'
        self.files = {name:('installer fixture '+name).encode() for name in ALLOWED}
        with zipfile.ZipFile(self.archive,'w') as z:
            for name,data in self.files.items(): z.writestr(name,data)
        self.manifest = {
            'archive_sha256':hashlib.sha256(self.archive.read_bytes()).hexdigest(),
            'release_url':'https://example.invalid/test-fixture',
            'files':{name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in self.files.items()}
        }
        self.target = self.root/'models/cat_lora'
    def tearDown(self): self.workspace.cleanup()
    def test_verified_install_is_idempotent(self):
        first = install_archive(self.archive,self.target,self.manifest)
        second = install_archive(self.archive,self.target,self.manifest)
        self.assertEqual(first,second)
        self.assertEqual(set(p.name for p in self.target.iterdir()),ALLOWED)
    def test_archive_corruption_is_rejected_before_install(self):
        self.archive.write_bytes(self.archive.read_bytes()+b'corruption')
        with self.assertRaisesRegex(ValueError,'checksum'): install_archive(self.archive,self.target,self.manifest)
        self.assertFalse(self.target.exists())
    def test_different_existing_release_is_preserved(self):
        self.target.mkdir(parents=True)
        existing = self.target/'MODEL_CARD.md'; existing.write_text('existing release')
        with self.assertRaises(FileExistsError): install_archive(self.archive,self.target,self.manifest)
        self.assertEqual(existing.read_text(),'existing release')
        self.assertEqual(len(list(self.target.iterdir())),1)
    def test_extra_path_is_rejected(self):
        with zipfile.ZipFile(self.archive,'a') as z: z.writestr('../escape.txt',b'not allowed')
        self.manifest['archive_sha256']=hashlib.sha256(self.archive.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'paths'): install_archive(self.archive,self.target,self.manifest)
        self.assertFalse((self.root/'escape.txt').exists())

if __name__=='__main__': unittest.main()

# Publishing checklist

Before creating the public GitHub repository:

- [ ] Review `README.md` and confirm the project description and credits.
- [ ] Review `THIRD_PARTY_NOTICES.md`.
- [ ] Choose and add a license for this repository. Do not publish copied code or assets from third-party projects unless their license permits it.
- [ ] Confirm that no game files, extracted models, textures, `IEVR Viewer.settings.json` files, executables or local paths are staged.
- [ ] Run `python -m py_compile g4tx_extractor.py`.
- [ ] Run `./build_exe.ps1` and test `dist/IEVR Viewer.exe` before making a release.

Suggested first commit:

```powershell
git add .
git status
git commit -m "Initial public release"
```

Then create an empty repository on GitHub and add its remote:

```powershell
git branch -M main
git remote add origin https://github.com/YOUR-ACCOUNT/IEVR-Viewer.git
git push -u origin main
```

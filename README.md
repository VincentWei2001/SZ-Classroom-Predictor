# SZ Classroom Predictor

Windows desktop app for building-performance prediction (Qt / PySide6, XGBoost / LightGBM / scikit-learn stack). Source lives in the Jupyter notebook and exported Python entrypoint; distribution is a PyInstaller folder bundle with an encrypted model archive.

## Download (end users)

**Do not rely on cloning this repo for the runnable app.** Install the latest **`school_app_secure_portable.zip`** from [Releases](https://github.com/VincentWei2001/SZ-Classroom-Predictor/releases).

1. Download the zip from Releases.
2. Extract to a folder (avoid paths that are only special characters).
3. Run `school_app.exe`.
4. Keep the `_internal` folder next to the exe; do not move the exe alone.

## Development

- **Notebook:** `预测应用完整版.ipynb`
- **Export portable script:**  
  `D:\Anaconda\envs\school_app\python.exe build_school_app.py`
- **Full secure build (exe + embedded `school_app_models.bin`):**  
  `D:\Anaconda\envs\school_app\python.exe build_secure_package.py`  
  Output zip at repo root; copy to `github_release/` if you use that layout for publishing.
- **Conda env:** Python 3.10 (`school_app`), packages as used in your local training stack (PySide6, xgboost, lightgbm, sklearn, etc.).

## Repository layout

- `build_secure_package.py` / `build_light_package.py` — packaging automation
- `secure_model_bundle.py` — builds `school_app_models.bin` (Fernet + zlib)
- `github_release/` — helper text for maintainers; the release zip is ignored by git (`*.zip`)

## Publishing to GitHub

Remote is expected to be `https://github.com/VincentWei2001/SZ-Classroom-Predictor.git`.

1. Commit and push: `git push -u origin main` (use a [PAT](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) if HTTPS prompts fail).
2. **Release zip:** upload `github_release/school_app_secure_portable.zip` on the [Releases](https://github.com/VincentWei2001/SZ-Classroom-Predictor/releases) page, or run (PowerShell, with `GITHUB_TOKEN` set):

   `.\scripts\publish_release.ps1`

If `git push` fails with “could not connect to github.com”, check VPN/proxy/firewall; the failure is environmental, not the repo layout.

## License

Add a `LICENSE` file if you publish publicly.

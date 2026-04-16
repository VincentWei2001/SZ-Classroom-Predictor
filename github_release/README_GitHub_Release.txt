GitHub Release 上传说明
========================

本目录中的 school_app_secure_portable.zip 为可直接分发的 Windows 便携包
（Python 3.10 / conda 环境 school_app 下由 build_secure_package.py 构建）。

用户侧使用
----------
1. 下载 zip（勿从仓库源码里找模型，以本附件为准）。
2. 解压到任意文件夹（路径中尽量避免仅含特殊字符）。
3. 双击 school_app.exe 启动。
4. 不要删除或移动 _internal 文件夹；勿单独挪走 exe。

发布者操作
----------
方式 A：在 GitHub 仓库页面 Releases -> Draft a new release，上传本目录中的
school_app_secure_portable.zip。

方式 B（自动化）：在仓库根目录 PowerShell 中设置 GITHUB_TOKEN（需 repo 权限），执行
  .\scripts\publish_release.ps1
会先 git push，再创建 Release 并上传 zip（默认 tag v1.0.0，可用环境变量 RELEASE_TAG 覆盖）。

若单文件超过 GitHub 附件大小上限，需拆包或使用外链，参见项目 README。

重新生成本 zip
--------------
在项目 CSV 目录下（已配置 Anaconda）依次执行：

  D:\Anaconda\envs\school_app\python.exe build_school_app.py
  D:\Anaconda\envs\school_app\python.exe build_secure_package.py

然后将根目录生成的 school_app_secure_portable.zip 复制到本 github_release 文件夹。

# 実行環境作成

Pythonは、割とライブラリのバージョン縛りがあったりして、複数のバージョンがインストールできた方が便利な場合があります。  
ということで、素のPythonではなく、Pythonの実行環境を管理するツールをインストールした上で、Python実行環境を用意します。  
ファイル容量は大きいですが、管理が楽なのでanacondaがおすすめです。


## anacondaを使う場合

anacondaのpythonを使う場合、vimのpython系プラグインを使おうとしたときにライブラリまわりで問題になる可能性があります。  
vim使いで、ライブラリパス等の扱いに詳しくない人は使わない方が、後々、変なことで悩まずに住むかもです。

1. python3.6実行環境をの作成
    ```bash
    $ conda create --name=py36 python=3.6
    ```

2. python3.6実行環境への切り替え
    - Linux、macOS
        ```bash
        $ source activate py36
        ```
    - Windows
        ```console
        $ activate py36
        ```

3. インストールしたAnacondaのメンテ
    * Anaconda自体のアップデート
        ```bash
        $ conda update -n base conda
        ```

    * Anacondaの全パッケージのアップデート
        ```bash
        $ conda update --all
        ```


## pyenv-virtualenvを使う場合

macOSではbrewコマンドでインストールしてできるので、簡単にインストールできます。  
Linuxだと、aptやyumでインストールできないので、インストールが面倒です。  
Linuxbrewを使えば楽かも知れない、と思いつつ試してはいません．．．

1. インストール
    - macOS
        ```bash
        $ brew install pyenv
        $ brew install pyenv-virtualenv
        ```
    - Linux  
pyenvとpyenv-virtualenvのインストール順には依存関係があるみたいです。
        - pyenvのインストール
            ```bash
            $ git clone https://github.com/pyenv/pyenv.git ~/.pyenv
            ```
        - $HOME/.bashrcへの追加 その1
            ```bash
            export PYENV_ROOT=$HOME/.pyenv
            export PATH=$PYENV_ROOT/bin:$PATH
            if command -v pyenv 1>/dev/null 2>&1; then
                eval "$(pyenv init -)"
            fi
            ```
        - $HOME/.bashrcの再読込み その1
            ```bash
            $ source $HOME/.bashrc
            ```

        - pyenv-virtualenvのインストール
            ```bash
            $ git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
            ```
        - $HOME/.bashrcへの追加 その2
            ```bash
            eval "$(pyenv virtualenv-init -)"
            ```
        - $HOME/.bashrcの再読込み その2
            ```bash
            $ source $HOME/.bashrc
            ```

2. python実行環境への切り替え
    - pythonビルドツールのインストール
        - CentOS
            ```bash
            $ sudo yum -y groupinstall "Development Tools"
            $ sudo yum -y install readline-devel \
                    zlib-devel bzip2-devel sqlite-devel openssl-devel
            ```
        - Debian系Linux
            ```bash
            $ sudo apt install -y gcc make \
                    libssl-dev libbz2-dev  \
                    libreadline-dev libsqlite3-dev zlib1g-dev
            ```

    - python3.6実行環境のインストール
        ```bash
        $ pyenv install 3.6.8
        $ pyenv virtualenv 3.6.8 mnist
        $ pyenv activate mnist
        ```


## virtualenvを使う場合

virtualenvの仮装環境は、pythonのバージョンが切り替えられないので、tensorflowのようにpythonのバージョン縛りがキツいライブラリ使いたい時には使えないことがあります。

```bash
$ virtualenv virt
$ source ./virt/bin/activate
$ pip install --upgrade tensorflow chainer sklearn opencv-python matplotlib numpy
```


# ライブラリのインストール

```bash
$ sudo apt-get install graphviz
$ pip install --upgrade \
    tensorflow keras chainer sklearn \
    opencv-python pillow matplotlib numpy \
    pydot-ng graphviz \
    jupyter flask
```

requirement.txtをgithubにおいておくと、ライブラリバージョンが抱える脆弱性を指摘するメールが鬱陶しいので、pipコマンドのみにしました。


# その他

## jupyter notebookの起動

```bash
$ jupyter notebook --config=./jupyter_nb_cfg.py --no-browser
```

を実行すると、ネットワーク上の他のPCからパスワードやトークン無しでアクセスできます。セキュリティ的にアレなので、火壁の内側だけで運用する場合以外は使用しない方が良いです。  
ちなみに、コンフィグファイルの拡張子は「.py」である必要があるっぽいです。


# よくある？問題

## macOSでmatplitlibとpylabを使おうとしたらRuntimeError

下記の内容で$HOME/.matplotlib/matplotlibrcを作成し、matplitlibの描画バックエンドを指定してあげれば直るようです。

```bash
backend : TkAgg
```


## pyenvの仮想環境でTkinterがimportできない

python環境インストール時にTkのビルドライブラリをインストールしていないと起こる現象らしいです。

```
$ sudo apt-get install tk-dev
$ pyenv install 3.6.8
```

のような感じで、Tkライブラリインストール後、python環境を新規インストールまたは上書きインストールするとimport出来るようになるはずです。


## dotコマンドが無い

Graphvizというアプリをインストールする必要があります。

* macOS
    ```
    $ brew install graphviz
    ・・・
    ==> libtool
    In order to prevent conflicts with Apple's own libtool we have prepended a "g"
    so, you have instead: glibtool and glibtoolize.
    ```
    libtoolのところでWarning？のようなメッセージが出ることが稀によくあるみたいですが、とりあえずdotコマンド使えたので深く考えないことにしています。

* Windows  
    [Graphviz - Graph Visualization Software](https://graphviz.gitlab.io/)のDownload → Stable x.xx Windows install packagesからインストーラーファイルをダウンロードできます。

# 注意事項

- pythonのバージョン  
2019年2月現在、たぶん、tensorflowがpython3.7以上に対応していないので、python3.6とかで実行すること。  
anacondaとかでpython環境作った方が管理が楽。

# 事項環境作成

最低限必要な環境作成例

```
$ virtualenv virt
$ source ./virt/bin/activate
$ pip install --upgrade tensorflow opencv-python matplotlib numpy
```

## jupyter notebook

1. インストール
    ```
    $ pip install --upgrade jupyter
    ```

2. 起動方法
    ```
    $ jupyter notebook --config=./jupyter_nb_cfg.py --no-browser
    ```
    jupyter notebook実行ホストからのみの利用であればオプションは不要だが、ネットワーク上の他のPCからもjupyter notebookを利用する場合、bindするIPアドレス変更しないとアクセスを受け付けてくれない。  
    あと、コンフィグファイルの拡張子は「.py」である必要があるっぽい。


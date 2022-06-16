#mnistの1を正常画像、0を異常画像
train.pyで学習させてresultフォルダにモデルを保存、
predict.pyで0と1の画像を含んだtestデータに対して、異常度を計算し、aucrocかaccuracyのスコアを表示

!python3 train.py -c config.json
で訓練
!python3 train.py -c config.json -t
でテスト


python test.py checkpoint --dataset Kodak -a cps-v4 \
-p experiment/cps_1_lambda00018_v4_best_loss.pth.tar \
-d output/cps_lambda00018_v4_patch256_Kodak \
--config configs/cps_1_lambda00018.yaml --save-image --per-image --patch --cuda

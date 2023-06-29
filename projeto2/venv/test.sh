#THRESHOLDING

rm -rf ./output_test/
mkdir ./output_test

#gourds_thresholding_t57.pgm
python3 main.py --imgpath ./input/gourds.pgm --op thresholding --t 57 --outputpath ./ > ./output_test/gourds_thresholding_t57.txt && mv ./thresholding.pgm ./output_test/gourds_thresholding_t57.pgm
cmp ./output_test/gourds_thresholding_t57.pgm ./output/gourds_thresholding_t57.pgm

#chest_thresholding_t100.pgm
python3 main.py --imgpath ./input/chest.pgm --op thresholding --t 100 --outputpath ./ > ./output_test/chest_thresholding_t100.txt && mv ./thresholding.pgm ./output_test/chest_thresholding_t100.pgm
cmp ./output_test/chest_thresholding_t100.pgm ./output/chest_thresholding_t100.pgm

#leaf_thresholding_t180.pgm
python3 main.py --imgpath ./input/leaf.pgm --op thresholding --t 180 --outputpath ./ > ./output_test/leaf_thresholding_t180.txt && mv ./thresholding.pgm ./output_test/leaf_thresholding_t180.pgm
cmp ./output_test/leaf_thresholding_t180.pgm ./output/leaf_thresholding_t180.pgm

#MEAN

#gourds_mean_k5.pgm
python3 main.py --imgpath ./input/gourds.pgm --op mean --k 5 --outputpath ./ > ./output_test/gourds_mean_k5.txt && mv ./mean.pgm ./output_test/gourds_mean_k5.pgm
cmp ./output_test/gourds_mean_k5.pgm ./output/gourds_mean_k5.pgm

#chest_mean_k3.pgm
python3 main.py --imgpath ./input/chest.pgm --op mean --k 3 --outputpath ./ > ./output_test/chest_mean_k3.txt && mv ./mean.pgm ./output_test/chest_mean_k3.pgm
cmp ./output_test/chest_mean_k3.pgm ./output/chest_mean_k3.pgm

#leaf_mean_k9.pgm
python3 main.py --imgpath ./input/leaf.pgm --op mean --k 9 --outputpath ./ > ./output_test/leaf_mean_k9.txt && mv ./mean.pgm ./output_test/leaf_mean_k9.pgm
cmp ./output_test/leaf_mean_k9.pgm ./output/leaf_mean_k9.pgm

#MEDIAN

#gourds_median_k13.pgm
python3 main.py --imgpath ./input/gourds.pgm --op median --k 13 --outputpath ./ > ./output_test/gourds_median_k13.txt && mv ./median.pgm ./output_test/gourds_median_k13.pgm
cmp ./output_test/gourds_median_k13.pgm ./output/gourds_median_k13.pgm

#chest_median_k7.pgm
python3 main.py --imgpath ./input/chest.pgm --op median --k 7 --outputpath ./ > ./output_test/chest_median_k7.txt && mv ./median.pgm ./output_test/chest_median_k7.pgm
cmp ./output_test/chest_median_k7.pgm ./output/chest_median_k7.pgm

#leaf_median_k9.pgm
python3 main.py --imgpath ./input/leaf.pgm --op median --k 9 --outputpath ./ > ./output_test/leaf_median_k9.txt && mv ./median.pgm ./output_test/leaf_median_k9.pgm
cmp ./output_test/leaf_median_k9.pgm ./output/leaf_median_k9.pgm

#SGT

#gourds_sgt_dt1.pgm / gourds_sgt_dt1.txt
python3 main.py --imgpath ./input/gourds.pgm --op sgt --dt 1 --outputpath ./ > ./output_test/gourds_sgt_dt1.txt && mv ./sgt.pgm ./output_test/gourds_sgt_dt1.pgm
cmp ./output_test/gourds_sgt_dt1.pgm ./output/gourds_sgt_dt1.pgm && cmp ./output_test/gourds_sgt_dt1.txt ./output/gourds_sgt_dt1.txt

#chest_sgt_dt1.pgm / chest_sgt_dt1.txt
python3 main.py --imgpath ./input/chest.pgm --op sgt --dt 1 --outputpath ./ > ./output_test/chest_sgt_dt1.txt && mv ./sgt.pgm ./output_test/chest_sgt_dt1.pgm
cmp ./output_test/chest_sgt_dt1.pgm ./output/chest_sgt_dt1.pgm && cmp ./output_test/chest_sgt_dt1.txt ./output/chest_sgt_dt1.txt

#leaf_sgt_dt1.pgm / leaf_sgt_dt1.txt
python3 main.py --imgpath ./input/leaf.pgm --op sgt --dt 1 --outputpath ./ > ./output_test/leaf_sgt_dt1.txt && mv ./sgt.pgm ./output_test/leaf_sgt_dt1.pgm
cmp ./output_test/leaf_sgt_dt1.pgm ./output/leaf_sgt_dt1.pgm && cmp ./output_test/leaf_sgt_dt1.txt ./output/leaf_sgt_dt1.txt
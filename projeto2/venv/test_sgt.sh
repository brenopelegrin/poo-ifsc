
rm -rf ./output_test/
mkdir ./output_test

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
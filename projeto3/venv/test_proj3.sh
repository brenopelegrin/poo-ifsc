rm -rf owl-mean-9-sobel mule-sobel-sgt-1 chicken-sobel
mkdir owl-mean-9-sobel mule-sobel-sgt-1 chicken-sobel

rm -rf input output
mkdir input output

cp -r input_proj3/* input/
cp -r output_proj3/* output

python main.py --imgpath ./input/chicken.pgm --op sobel --outputpath ./chicken-sobel
python main.py --imgpath ./input/mule.pgm --op sobel sgt --dt 1 --outputpath ./mule-sobel-sgt-1
python main.py --imgpath ./input/owl.pgm --op mean --k 9 sobel --outputpath ./owl-mean-9-sobel

diff -q ./output/chicken-sobel.pgm ./chicken-sobel/output.pgm
diff -q ./output/mule-sobel-sgt-1.pgm ./mule-sobel-sgt-1/output.pgm
diff -q ./output/owl-mean-9-sobel.pgm ./owl-mean-9-sobel/output.pgm

rm -rf input output
rm -rf owl-mean-9-sobel mule-sobel-sgt-1 chicken-sobel



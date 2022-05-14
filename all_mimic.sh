1. CUDA_VISIBLE_DEVICES=0  python test_learning.py -epoch 100 -model 'HCL' -save-label 'HCL(event only)' -data-folder 'tpp-data/data_retweet' -epoch 100 -w-mle 0 -w-dis 1 -w-cl1 0.1 -w-cl2 0

2. CUDA_VISIBLE_DEVICES=1  python test_learning.py -epoch 100 -model 'HCL' -save-label 'HCL(event only)' -data-folder 'tpp-data/data_retweet' -epoch 100 -w-mle 0 -w-dis 1 -w-cl1 1 -w-cl2 0

3. CUDA_VISIBLE_DEVICES=1  python test_learning.py -epoch 100 -model 'HCL' -save-label 'HCL(event only)' -data-folder 'tpp-data/data_retweet' -epoch 100 -w-mle 0 -w-dis 1 -w-cl1 10 -w-cl2 0

4. CUDA_VISIBLE_DEVICES=0  python test_learning.py -epoch 100 -model 'HCL' -save-label 'HCL(seq only)' -data-folder 'tpp-data/data_retweet' -epoch 100 -w-mle 0 -w-dis 1 -w-cl1 0 -w-cl2 0.1

5. CUDA_VISIBLE_DEVICES=2  python test_learning.py -epoch 100 -model 'HCL' -save-label 'HCL(seq only)' -data-folder 'tpp-data/data_retweet' -epoch 100 -w-mle 0 -w-dis 1 -w-cl1 0 -w-cl2 1

6. CUDA_VISIBLE_DEVICES=2  python test_learning.py -epoch 100 -model 'HCL' -save-label 'HCL(seq only)' -data-folder 'tpp-data/data_retweet' -epoch 100 -w-mle 0 -w-dis 1 -w-cl1 0 -w-cl2 10

7. CUDA_VISIBLE_DEVICES=0  python test_learning.py  -epoch 100 -model 'HCL' -save-label 'HCL(both)' -data-folder 'tpp-data/data_retweet' -epoch 100 -w-mle 0 -w-dis 1 -w-cl1 0.1 -w-cl2 0.1

8. CUDA_VISIBLE_DEVICES=1  python test_learning.py  -epoch 100 -model 'HCL' -save-label 'HCL(both)' -data-folder 'tpp-data/data_retweet' -epoch 100 -w-mle 0 -w-dis 1 -w-cl1 1 -w-cl2 1

9. CUDA_VISIBLE_DEVICES=2  python test_learning.py  -epoch 100 -model 'HCL' -save-label 'HCL(both)' -data-folder 'tpp-data/data_retweet' -epoch 100 -w-mle 0 -w-dis 1 -w-cl1 10 -w-cl2 10

10. CUDA_VISIBLE_DEVICES=0  python test_learning.py  -epoch 100 -model 'HCL' -save-label 'HCL(both)' -data-folder 'tpp-data/data_retweet' -epoch 100 -w-mle 0 -w-dis 1 -w-cl1 0.1 -w-cl2 10

11. CUDA_VISIBLE_DEVICES=1  python test_learning.py  -epoch 100 -model 'HCL' -save-label 'HCL(both)' -data-folder 'tpp-data/data_retweet' -epoch 100 -w-mle 0 -w-dis 1 -w-cl1 10 -w-cl2 1

CUDA_VISIBLE_DEVICES=2  python test_learning.py  -epoch 100 -model 'HCL' -save-label 'HCL(both)' -data-folder 'tpp-data/data_retweet' -epoch 100 -w-mle 0 -w-dis 1 -w-cl1 1 -w-cl2 10
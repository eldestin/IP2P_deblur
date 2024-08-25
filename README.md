## Instruction tuning of Instructpix2pix and stable diffusion v1.5
This is the repository of our project: instruction tuning based defocus deblurring.


## Download the training and evaluation datasets
[DPDD dataset](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel)

[RealDof dataset](https://github.com/codeslake/IFAN)

[CUHK dataset](https://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/dataset.html)

## Training
```
python main_{dataset}.py
```

## Evaluation
First please change the validation path in each .py file. Then run:
```
python main_{dataset}.py
```
## Experimental result
The experiment result and comparison is still in progress.

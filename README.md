# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py project/run_fast_tensor.py project/parallel_check.py tests/test_tensor_general.py

## `run_sentiment.py`
```
missing pre-trained embedding for 55 unknown words
Epoch 1, loss 31.55183703174403, train accuracy: 50.22%
Validation accuracy: 57.00%
Best Valid accuracy: 57.00%
Epoch 2, loss 31.362306318853882, train accuracy: 51.11%
Validation accuracy: 53.00%
Best Valid accuracy: 57.00%
Epoch 3, loss 31.12041768562897, train accuracy: 54.89%
Validation accuracy: 60.00%
Best Valid accuracy: 60.00%
Epoch 4, loss 30.947174437499918, train accuracy: 54.44%
Validation accuracy: 50.00%
Best Valid accuracy: 60.00%
Epoch 5, loss 30.839089693737716, train accuracy: 55.56%
Validation accuracy: 58.00%
Best Valid accuracy: 60.00%
Epoch 6, loss 30.5483646586799, train accuracy: 60.22%
Validation accuracy: 60.00%
Best Valid accuracy: 60.00%
Epoch 7, loss 30.591952531250055, train accuracy: 57.11%
Validation accuracy: 62.00%
Best Valid accuracy: 62.00%
Epoch 8, loss 30.14499680423576, train accuracy: 62.89%
Validation accuracy: 60.00%
Best Valid accuracy: 62.00%
Epoch 9, loss 30.308680924945726, train accuracy: 59.78%
Validation accuracy: 61.00%
Best Valid accuracy: 62.00%
Epoch 10, loss 29.848891292207618, train accuracy: 62.67%
Validation accuracy: 63.00%
Best Valid accuracy: 63.00%
Epoch 11, loss 29.521885890564693, train accuracy: 67.78%
Validation accuracy: 65.00%
Best Valid accuracy: 65.00%
Epoch 12, loss 29.00158457587541, train accuracy: 66.89%
Validation accuracy: 63.00%
Best Valid accuracy: 65.00%
Epoch 13, loss 28.738740961749915, train accuracy: 70.00%
Validation accuracy: 68.00%
Best Valid accuracy: 68.00%
Epoch 14, loss 28.54537537683169, train accuracy: 69.33%
Validation accuracy: 65.00%
Best Valid accuracy: 68.00%
Epoch 15, loss 28.15425362623081, train accuracy: 72.00%
Validation accuracy: 67.00%
Best Valid accuracy: 68.00%
Epoch 16, loss 27.602667206511548, train accuracy: 71.78%
Validation accuracy: 67.00%
Best Valid accuracy: 68.00%
Epoch 17, loss 26.732699498448387, train accuracy: 74.22%
Validation accuracy: 70.00%
Best Valid accuracy: 70.00%
Epoch 18, loss 26.188691275795065, train accuracy: 75.56%
Validation accuracy: 70.00%
Best Valid accuracy: 70.00%
Epoch 19, loss 25.48419282249229, train accuracy: 77.56%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 20, loss 24.69583613801053, train accuracy: 78.89%
Validation accuracy: 69.00%
Best Valid accuracy: 71.00%
Epoch 21, loss 24.48176275379217, train accuracy: 76.44%
Validation accuracy: 72.00%
Best Valid accuracy: 72.00%
Epoch 22, loss 24.04217912363848, train accuracy: 79.56%
Validation accuracy: 74.00%
Best Valid accuracy: 74.00%
Epoch 23, loss 23.012211267223456, train accuracy: 80.67%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 24, loss 22.185434131149705, train accuracy: 81.33%
Validation accuracy: 74.00%
Best Valid accuracy: 74.00%
Epoch 25, loss 22.175869257474872, train accuracy: 80.44%
Validation accuracy: 74.00%
Best Valid accuracy: 74.00%
Epoch 26, loss 21.20578679024366, train accuracy: 81.33%
Validation accuracy: 74.00%
Best Valid accuracy: 74.00%
Epoch 27, loss 20.02447495036669, train accuracy: 83.33%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 28, loss 19.669951058568675, train accuracy: 83.56%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 29, loss 19.26919096294848, train accuracy: 83.56%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 30, loss 18.3229700305106, train accuracy: 84.67%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 31, loss 17.93180092512894, train accuracy: 85.56%
Validation accuracy: 76.00%
Best Valid accuracy: 76.00%
Epoch 32, loss 17.24040438713265, train accuracy: 87.11%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 33, loss 16.552671019426192, train accuracy: 87.11%
Validation accuracy: 76.00%
Best Valid accuracy: 76.00%
Epoch 34, loss 15.539536701772036, train accuracy: 89.33%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 35, loss 15.551426725578917, train accuracy: 88.89%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 36, loss 15.110204309771285, train accuracy: 89.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 37, loss 14.749487815552165, train accuracy: 88.89%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 38, loss 14.335843241780285, train accuracy: 89.11%
Validation accuracy: 76.00%
Best Valid accuracy: 76.00%
Epoch 39, loss 13.703455872761669, train accuracy: 91.56%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 40, loss 13.044080037037595, train accuracy: 90.67%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 41, loss 12.45549246402747, train accuracy: 92.67%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 42, loss 12.131505659896687, train accuracy: 92.67%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 43, loss 11.803769756915996, train accuracy: 92.67%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 44, loss 10.732635316285378, train accuracy: 94.00%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 45, loss 11.663638234303022, train accuracy: 91.11%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 46, loss 10.237282328103692, train accuracy: 94.67%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 47, loss 10.016586245559857, train accuracy: 95.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 48, loss 10.40667586386685, train accuracy: 94.44%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 49, loss 9.84575102531962, train accuracy: 95.56%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 50, loss 9.728414558374627, train accuracy: 96.00%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 51, loss 9.0765099624323, train accuracy: 94.89%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 52, loss 9.053592085905475, train accuracy: 94.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 53, loss 8.544647388750832, train accuracy: 95.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 54, loss 7.955620006074709, train accuracy: 96.89%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 55, loss 8.027001110078965, train accuracy: 97.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 56, loss 7.502406609229805, train accuracy: 96.67%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 57, loss 7.729122081257757, train accuracy: 96.22%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 58, loss 7.427072758855148, train accuracy: 97.33%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 59, loss 7.211434547960011, train accuracy: 96.89%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 60, loss 6.954337032021038, train accuracy: 97.78%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 61, loss 6.633299257923544, train accuracy: 98.22%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 62, loss 6.578540554619366, train accuracy: 97.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 63, loss 5.93587521684019, train accuracy: 98.00%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 64, loss 5.978351085149053, train accuracy: 98.44%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 65, loss 6.101567825770901, train accuracy: 96.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 66, loss 5.931783106223301, train accuracy: 98.22%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 67, loss 5.28949679043701, train accuracy: 98.67%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 68, loss 5.359867565323205, train accuracy: 98.22%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 69, loss 5.537456137818607, train accuracy: 97.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 70, loss 5.324594291513718, train accuracy: 99.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 71, loss 5.2648585822324, train accuracy: 98.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 72, loss 4.7781827076722445, train accuracy: 99.11%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 73, loss 5.367136066639553, train accuracy: 97.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 74, loss 4.814556018630749, train accuracy: 99.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 75, loss 5.1863259742188665, train accuracy: 97.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 76, loss 4.53092401185546, train accuracy: 99.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 77, loss 4.543067444092181, train accuracy: 98.89%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 78, loss 4.6137176394245945, train accuracy: 98.67%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 79, loss 4.5725820661500025, train accuracy: 98.22%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 80, loss 4.5605926514365445, train accuracy: 98.67%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 81, loss 4.570760933586769, train accuracy: 98.89%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 82, loss 4.110550036789425, train accuracy: 99.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 83, loss 3.661521431615864, train accuracy: 98.89%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 84, loss 3.924862856725704, train accuracy: 99.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 85, loss 3.9420023288887234, train accuracy: 99.33%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 86, loss 3.553468866290651, train accuracy: 99.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 87, loss 4.016792760480369, train accuracy: 98.67%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 88, loss 3.7021777035555576, train accuracy: 98.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 89, loss 3.6930004232387668, train accuracy: 99.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 90, loss 3.67065180825475, train accuracy: 98.89%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 91, loss 3.7698295960636723, train accuracy: 98.89%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 92, loss 3.536415581236613, train accuracy: 99.33%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 93, loss 3.119194436690017, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 94, loss 3.2959171477321934, train accuracy: 99.33%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 95, loss 3.1381676583651954, train accuracy: 99.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 96, loss 3.144211957689494, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 97, loss 2.8231488235993583, train accuracy: 99.56%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 98, loss 3.0228062239100573, train accuracy: 99.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 99, loss 3.5403321901001816, train accuracy: 99.11%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 100, loss 3.0059852431165517, train accuracy: 99.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 101, loss 3.219646818651718, train accuracy: 98.89%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 102, loss 2.844522922938823, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 103, loss 2.8237665132816003, train accuracy: 99.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 104, loss 2.6756151530451575, train accuracy: 99.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 105, loss 2.6754309476821883, train accuracy: 99.33%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 106, loss 2.8321507459130846, train accuracy: 99.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 107, loss 2.7001996356696605, train accuracy: 99.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 108, loss 2.68177492194305, train accuracy: 99.33%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 109, loss 2.4325620608673835, train accuracy: 99.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 110, loss 2.3366549323946666, train accuracy: 99.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 111, loss 2.898111549899109, train accuracy: 98.67%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 112, loss 2.3123491196623274, train accuracy: 99.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 113, loss 2.435853930249045, train accuracy: 99.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 114, loss 2.238170735363716, train accuracy: 99.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 115, loss 2.4178472574552727, train accuracy: 99.56%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 116, loss 2.3006159976098166, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 117, loss 2.275481679744503, train accuracy: 99.56%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 118, loss 2.2767054229054935, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 119, loss 2.242891559345618, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 120, loss 1.9342186080038466, train accuracy: 99.56%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 121, loss 1.9834682533711232, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 122, loss 2.1138660212945224, train accuracy: 99.78%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 123, loss 2.4157861647679266, train accuracy: 99.11%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 124, loss 2.099289751528504, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 125, loss 2.2143013012678883, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 126, loss 1.918845617767719, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 127, loss 1.855712786776037, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 128, loss 1.981186165383669, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 129, loss 1.7290507581534293, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 130, loss 1.9016724856138405, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 131, loss 1.7159932847149748, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 132, loss 1.7713939825083704, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 133, loss 1.890722408186998, train accuracy: 99.56%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 134, loss 1.8923146936331627, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 135, loss 2.083420803691067, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 136, loss 2.0108992252717335, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 137, loss 1.7997268653199563, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 138, loss 1.653685679838422, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 139, loss 1.8158107802259096, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 140, loss 1.8282503017997809, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 141, loss 1.725676934753455, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 142, loss 1.8057505686090236, train accuracy: 99.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 143, loss 1.6815892741062441, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 144, loss 1.5914536178593155, train accuracy: 99.78%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 145, loss 1.6763319760404718, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 146, loss 1.9235028071013847, train accuracy: 99.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 147, loss 1.610989842524714, train accuracy: 99.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 148, loss 1.7925964844810953, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 149, loss 1.8084082072736876, train accuracy: 99.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 150, loss 1.6455656770784075, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 151, loss 1.9621435281366204, train accuracy: 98.89%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 152, loss 1.52781859532811, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 153, loss 1.7629707328747344, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 154, loss 1.561816930391377, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 155, loss 1.5808363545557145, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 156, loss 1.5569968348195804, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 157, loss 1.5179884673954351, train accuracy: 99.33%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 158, loss 1.3636811044720518, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 159, loss 1.3694948815895436, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 160, loss 1.270718885730565, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 161, loss 1.5501691941495626, train accuracy: 99.33%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 162, loss 1.2509844757331108, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 163, loss 1.3536314984249103, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 164, loss 1.3083672438217504, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 165, loss 1.6601978336409136, train accuracy: 99.33%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 166, loss 1.4316737341608117, train accuracy: 99.56%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 167, loss 1.2978903939915223, train accuracy: 99.56%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 168, loss 1.2776914704968716, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 169, loss 1.2689781531527904, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 170, loss 1.3527071016779826, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 171, loss 1.2214332727492008, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 172, loss 1.4236056191185615, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 173, loss 1.2701453964521794, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 174, loss 1.3816604076879015, train accuracy: 99.56%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 175, loss 1.3507359305042181, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 176, loss 1.177253161744344, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 177, loss 1.1497675081137677, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 178, loss 1.2635247981980327, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 179, loss 1.4945413671359127, train accuracy: 99.33%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 180, loss 1.1578461155451083, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 181, loss 1.2200617729212824, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 182, loss 1.1487575781884407, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 183, loss 1.165025168457032, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 184, loss 1.21891455438, train accuracy: 99.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 185, loss 1.0968067636779113, train accuracy: 99.56%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 186, loss 1.2003994370050821, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 187, loss 1.108786637011593, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 188, loss 1.0876601038311065, train accuracy: 99.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 189, loss 1.1068805957638281, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 190, loss 1.2175462044823668, train accuracy: 99.56%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 191, loss 1.2107329118634453, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 192, loss 0.976169681933946, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 193, loss 1.175634318071478, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 194, loss 1.0463633663580325, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 195, loss 1.065132859057309, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 196, loss 1.0716012716541885, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 197, loss 1.343477093324486, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 198, loss 1.0399581252374999, train accuracy: 99.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 199, loss 1.275145080247698, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 200, loss 0.9519428330819724, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 201, loss 0.9864134807267295, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 202, loss 1.1803038713540657, train accuracy: 99.56%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 203, loss 0.9738964373263276, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 204, loss 1.2268122750117008, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 205, loss 0.9545165686218933, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 206, loss 0.909476450931154, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 207, loss 0.9321850666931215, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 208, loss 1.1417173154092712, train accuracy: 99.56%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 209, loss 1.06058892830696, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 210, loss 0.9441876689286486, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 211, loss 1.0231984863200336, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 212, loss 0.9783634931319557, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 213, loss 0.9215982506834322, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 214, loss 1.0539411246581116, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 215, loss 0.934490971227735, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 216, loss 0.8710627255736192, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 217, loss 0.7966996392402711, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 218, loss 0.9693961967231763, train accuracy: 99.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 219, loss 0.946826184843981, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 220, loss 0.9044013346475372, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 221, loss 1.0826801689856915, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 222, loss 0.9884511469583832, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 223, loss 0.9273947406229616, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 224, loss 0.8017798317758178, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 225, loss 0.8786676566300679, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 226, loss 1.0391526804903668, train accuracy: 99.56%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 227, loss 0.9897267227644002, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 228, loss 1.050749154269334, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 229, loss 0.9775370244378323, train accuracy: 99.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 230, loss 1.0944835817269667, train accuracy: 99.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 231, loss 1.0123611739413474, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 232, loss 0.8878167911395121, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 233, loss 0.8338997213439899, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 234, loss 0.904327021697719, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 235, loss 0.8772320533574698, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 236, loss 0.8584754141953961, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 237, loss 0.9204461924046337, train accuracy: 99.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 238, loss 0.9381697710975796, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 239, loss 0.9000690378252922, train accuracy: 99.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 240, loss 0.8299831344368934, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 241, loss 1.0367335400945243, train accuracy: 99.56%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 242, loss 0.8474534988767575, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 243, loss 0.9143726800202616, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 244, loss 0.7566977635413494, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 245, loss 1.2282599964431549, train accuracy: 99.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 246, loss 1.020824060482217, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 247, loss 0.9363135539825961, train accuracy: 99.56%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 248, loss 0.6859830405965982, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 249, loss 0.9355002428729557, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 250, loss 0.7204180510004441, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%

real    17m50,917s
user    88m59,154s
sys     1m30,035s
```

### CUDA
```
missing pre-trained embedding for 55 unknown words
USE_CUDA_CONV
Epoch 1, loss 31.516165919867163, train accuracy: 50.22%
Validation accuracy: 49.00%
Best Valid accuracy: 49.00%
Epoch 2, loss 31.147699862750716, train accuracy: 52.67%
Validation accuracy: 48.00%
Best Valid accuracy: 49.00%
Epoch 3, loss 30.99902567897589, train accuracy: 54.22%
Validation accuracy: 47.00%
Best Valid accuracy: 49.00%
Epoch 4, loss 30.731888300528468, train accuracy: 57.33%
Validation accuracy: 58.00%
Best Valid accuracy: 58.00%
Epoch 5, loss 30.731230479319915, train accuracy: 57.33%
Validation accuracy: 54.00%
Best Valid accuracy: 58.00%
Epoch 6, loss 30.29736775011337, train accuracy: 61.11%
Validation accuracy: 57.00%
Best Valid accuracy: 58.00%
Epoch 7, loss 30.189806824056344, train accuracy: 61.11%
Validation accuracy: 58.00%
Best Valid accuracy: 58.00%
Epoch 8, loss 29.842374885482634, train accuracy: 64.44%
Validation accuracy: 58.00%
Best Valid accuracy: 58.00%
Epoch 9, loss 29.72574214700065, train accuracy: 64.89%
Validation accuracy: 63.00%
Best Valid accuracy: 63.00%
Epoch 10, loss 29.122508045078746, train accuracy: 67.11%
Validation accuracy: 62.00%
Best Valid accuracy: 63.00%
Epoch 11, loss 28.793894124776553, train accuracy: 67.33%
Validation accuracy: 64.00%
Best Valid accuracy: 64.00%
Epoch 12, loss 28.396603297231767, train accuracy: 69.56%
Validation accuracy: 64.00%
Best Valid accuracy: 64.00%
Epoch 13, loss 27.871287108609682, train accuracy: 72.44%
Validation accuracy: 64.00%
Best Valid accuracy: 64.00%
Epoch 14, loss 27.404093452529168, train accuracy: 75.11%
Validation accuracy: 68.00%
Best Valid accuracy: 68.00%
Epoch 15, loss 26.66707862326922, train accuracy: 76.67%
Validation accuracy: 68.00%
Best Valid accuracy: 68.00%
Epoch 16, loss 26.237883219084008, train accuracy: 75.78%
Validation accuracy: 69.00%
Best Valid accuracy: 69.00%
Epoch 17, loss 25.3606976706591, train accuracy: 76.67%
Validation accuracy: 69.00%
Best Valid accuracy: 69.00%
Epoch 18, loss 24.852865620619742, train accuracy: 76.44%
Validation accuracy: 70.00%
Best Valid accuracy: 70.00%
Epoch 19, loss 24.138221640395997, train accuracy: 78.67%
Validation accuracy: 69.00%
Best Valid accuracy: 70.00%
Epoch 20, loss 23.594189670309934, train accuracy: 79.33%
Validation accuracy: 69.00%
Best Valid accuracy: 70.00%
Epoch 21, loss 23.286345207525617, train accuracy: 79.33%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 22, loss 22.248782100542023, train accuracy: 80.44%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 23, loss 21.977493939148747, train accuracy: 80.89%
Validation accuracy: 70.00%
Best Valid accuracy: 71.00%
Epoch 24, loss 21.02677183485721, train accuracy: 81.33%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 25, loss 20.080472313435507, train accuracy: 84.44%
Validation accuracy: 72.00%
Best Valid accuracy: 72.00%
Epoch 26, loss 20.003533655682862, train accuracy: 82.22%
Validation accuracy: 72.00%
Best Valid accuracy: 72.00%
Epoch 27, loss 18.605806297935708, train accuracy: 86.22%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%
Epoch 28, loss 18.18609446511184, train accuracy: 84.89%
Validation accuracy: 72.00%
Best Valid accuracy: 73.00%
Epoch 29, loss 18.16031516536177, train accuracy: 83.56%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%
Epoch 30, loss 17.112125523843197, train accuracy: 87.33%
Validation accuracy: 74.00%
Best Valid accuracy: 74.00%
Epoch 31, loss 16.39113480166007, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 32, loss 16.339188342433562, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 33, loss 15.33007841825647, train accuracy: 89.56%
Validation accuracy: 74.00%
Best Valid accuracy: 74.00%
Epoch 34, loss 15.238714688021718, train accuracy: 87.33%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 35, loss 14.326986705629446, train accuracy: 89.78%
Validation accuracy: 74.00%
Best Valid accuracy: 74.00%
Epoch 36, loss 13.91961987458981, train accuracy: 89.56%
Validation accuracy: 74.00%
Best Valid accuracy: 74.00%
Epoch 37, loss 12.750591678747947, train accuracy: 92.00%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 38, loss 13.08904186241088, train accuracy: 91.11%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 39, loss 12.953299923156047, train accuracy: 91.11%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 40, loss 12.749016714776165, train accuracy: 92.22%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 41, loss 11.676862640882206, train accuracy: 92.89%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 42, loss 11.11316526709126, train accuracy: 92.67%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 43, loss 11.14236856089385, train accuracy: 94.67%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 44, loss 10.630811482297286, train accuracy: 93.78%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 45, loss 10.259442051961688, train accuracy: 92.89%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 46, loss 10.351959728205202, train accuracy: 94.00%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 47, loss 9.696895865833538, train accuracy: 96.44%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 48, loss 9.526677987909055, train accuracy: 94.89%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 49, loss 9.01420866811451, train accuracy: 96.00%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 50, loss 9.060390420911208, train accuracy: 96.89%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 51, loss 8.660426008258847, train accuracy: 96.89%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 52, loss 8.245300801401939, train accuracy: 96.89%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 53, loss 7.9359926957186575, train accuracy: 97.33%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 54, loss 7.601324443850482, train accuracy: 97.33%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 55, loss 7.627641474607432, train accuracy: 96.44%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 56, loss 7.430385960404836, train accuracy: 97.56%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 57, loss 7.119626993147843, train accuracy: 96.89%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 58, loss 7.0557907232493555, train accuracy: 97.78%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 59, loss 6.269529447693063, train accuracy: 98.44%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 60, loss 6.545201806261055, train accuracy: 98.22%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 61, loss 6.673125085530512, train accuracy: 97.33%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 62, loss 6.685237121344161, train accuracy: 97.11%
Validation accuracy: 67.00%
Best Valid accuracy: 75.00%
Epoch 63, loss 6.541271984387997, train accuracy: 97.33%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 64, loss 5.880672600750754, train accuracy: 98.67%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 65, loss 5.8581341334080745, train accuracy: 98.22%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 66, loss 6.2089457768941, train accuracy: 98.00%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 67, loss 5.3323422258567295, train accuracy: 98.22%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 68, loss 5.7149105284240695, train accuracy: 98.44%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 69, loss 5.267994058512232, train accuracy: 98.22%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 70, loss 4.702212105836528, train accuracy: 98.89%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 71, loss 4.850263537478506, train accuracy: 99.11%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 72, loss 4.851446534612693, train accuracy: 98.89%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 73, loss 4.868407378497159, train accuracy: 98.67%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 74, loss 4.6174484416743615, train accuracy: 98.44%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 75, loss 4.5063495062785925, train accuracy: 99.11%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 76, loss 4.256667052665606, train accuracy: 98.89%
Validation accuracy: 67.00%
Best Valid accuracy: 75.00%
Epoch 77, loss 4.501924259470273, train accuracy: 98.44%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 78, loss 4.050283716415754, train accuracy: 98.89%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 79, loss 3.9146921607556253, train accuracy: 98.89%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 80, loss 3.9595069003578462, train accuracy: 98.89%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 81, loss 3.950712074703787, train accuracy: 98.44%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 82, loss 4.097755009863775, train accuracy: 98.89%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 83, loss 4.0548942584841825, train accuracy: 98.89%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 84, loss 3.567421558713075, train accuracy: 99.33%
Validation accuracy: 67.00%
Best Valid accuracy: 75.00%
Epoch 85, loss 3.9097153242245946, train accuracy: 99.11%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 86, loss 3.5921084510603722, train accuracy: 99.11%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 87, loss 3.204180670735899, train accuracy: 98.89%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 88, loss 3.691523962204873, train accuracy: 98.89%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 89, loss 3.571873815684509, train accuracy: 99.56%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 90, loss 3.114432770363618, train accuracy: 99.33%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 91, loss 3.479077183153302, train accuracy: 98.89%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 92, loss 3.2234905027147227, train accuracy: 99.11%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 93, loss 3.4805817776299923, train accuracy: 98.89%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 94, loss 3.250024296010126, train accuracy: 99.33%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 95, loss 3.304096013549996, train accuracy: 98.89%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 96, loss 2.9971615384036725, train accuracy: 98.89%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 97, loss 3.0803782978750087, train accuracy: 98.89%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 98, loss 2.9409250068022503, train accuracy: 99.11%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 99, loss 2.953260593689138, train accuracy: 98.89%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 100, loss 2.844356818442201, train accuracy: 99.33%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 101, loss 2.6657456301870965, train accuracy: 99.56%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 102, loss 2.724367153705808, train accuracy: 99.33%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 103, loss 2.7056286415197754, train accuracy: 99.56%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 104, loss 2.587128864980016, train accuracy: 99.33%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 105, loss 2.515866252538679, train accuracy: 99.33%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 106, loss 2.517145615910286, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 107, loss 2.561460343830798, train accuracy: 98.89%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 108, loss 2.670289974222452, train accuracy: 99.33%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 109, loss 2.4388858059480323, train accuracy: 99.33%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 110, loss 2.6316248046188977, train accuracy: 99.11%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 111, loss 2.6240082229213675, train accuracy: 99.11%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 112, loss 2.3114984013265705, train accuracy: 99.56%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 113, loss 2.278079084306792, train accuracy: 99.56%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 114, loss 2.2614236365107625, train accuracy: 99.33%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 115, loss 2.600240543594559, train accuracy: 99.11%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 116, loss 2.5798999128804416, train accuracy: 99.11%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 117, loss 1.983435093432703, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 118, loss 2.055837703068371, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 119, loss 2.2692057602676905, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 120, loss 2.038427339415553, train accuracy: 99.56%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 121, loss 2.1170747789935307, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 122, loss 2.1119500062533563, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 123, loss 1.8999819263876312, train accuracy: 99.78%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 124, loss 1.8701546759211498, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 125, loss 2.3069778510007555, train accuracy: 99.33%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 126, loss 2.1295360844513724, train accuracy: 99.33%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 127, loss 1.866299508437756, train accuracy: 99.33%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 128, loss 1.9219860941258844, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 129, loss 1.883554147975013, train accuracy: 99.56%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 130, loss 1.9860353906074266, train accuracy: 99.33%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 131, loss 1.9377326786194888, train accuracy: 99.56%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 132, loss 1.7113406744031203, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 133, loss 1.8366632813484534, train accuracy: 99.56%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 134, loss 1.8078060400693396, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 135, loss 1.684529266306231, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 136, loss 1.8194377324771616, train accuracy: 98.89%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 137, loss 1.6778026151283645, train accuracy: 99.56%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 138, loss 1.8187886218141982, train accuracy: 99.56%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 139, loss 1.7024785518439254, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 140, loss 1.7736631635123232, train accuracy: 99.33%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 141, loss 1.7618225516189137, train accuracy: 99.56%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 142, loss 1.6837642908730701, train accuracy: 99.56%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 143, loss 1.972140212951782, train accuracy: 99.56%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 144, loss 1.3509251764491155, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 145, loss 1.5691630308674023, train accuracy: 99.78%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 146, loss 1.471082526359868, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 147, loss 1.8269502446053163, train accuracy: 99.11%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 148, loss 1.4782914065570851, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 149, loss 1.4668290167448295, train accuracy: 99.78%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 150, loss 1.6611102693526194, train accuracy: 99.56%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 151, loss 1.8333876097975252, train accuracy: 99.56%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 152, loss 1.4682697243843623, train accuracy: 99.56%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 153, loss 1.8513092910343398, train accuracy: 99.33%
Validation accuracy: 67.00%
Best Valid accuracy: 75.00%
Epoch 154, loss 1.5015976702735687, train accuracy: 99.33%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 155, loss 1.455859487128115, train accuracy: 99.56%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 156, loss 1.4313782468334288, train accuracy: 99.78%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 157, loss 1.6263617415357265, train accuracy: 99.33%
Validation accuracy: 67.00%
Best Valid accuracy: 75.00%
Epoch 158, loss 1.6005647962634733, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 159, loss 1.436177953590955, train accuracy: 99.33%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 160, loss 1.5244822390550374, train accuracy: 99.56%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 161, loss 1.7196435867606406, train accuracy: 99.33%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 162, loss 1.3249213234645483, train accuracy: 99.56%
Validation accuracy: 67.00%
Best Valid accuracy: 75.00%
Epoch 163, loss 1.4090371071996595, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 164, loss 1.4915483298791499, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 165, loss 1.3854127333846948, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 166, loss 1.4985875983188157, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 167, loss 1.2900221797001714, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 168, loss 1.2334446568472472, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 169, loss 1.2836840091355075, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 170, loss 1.3837441870346525, train accuracy: 99.56%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 171, loss 1.3530425233376828, train accuracy: 99.56%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 172, loss 1.1598916049752777, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 173, loss 1.1802177245825074, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 174, loss 1.2333628697317973, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 175, loss 1.2596294552059062, train accuracy: 99.56%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 176, loss 1.384596070121108, train accuracy: 99.56%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 177, loss 1.1755660314318142, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 178, loss 1.5692693829733144, train accuracy: 99.11%
Validation accuracy: 67.00%
Best Valid accuracy: 75.00%
Epoch 179, loss 1.1364284782741871, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 180, loss 1.0947553627031121, train accuracy: 99.78%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 181, loss 1.1212756977183127, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 182, loss 1.1752168534320482, train accuracy: 99.33%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 183, loss 1.1906422136099872, train accuracy: 99.56%
Validation accuracy: 67.00%
Best Valid accuracy: 75.00%
Epoch 184, loss 1.197413431206235, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 185, loss 1.2259849858381655, train accuracy: 99.78%
Validation accuracy: 67.00%
Best Valid accuracy: 75.00%
Epoch 186, loss 1.0759554272260001, train accuracy: 99.56%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 187, loss 0.9688152853620975, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 188, loss 1.1144386113663645, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 189, loss 1.2349520703114134, train accuracy: 99.56%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 190, loss 1.1117304148966858, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 191, loss 1.1596272568382893, train accuracy: 99.78%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 192, loss 0.946020419010151, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 193, loss 1.0237558227007206, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 194, loss 1.101750183086962, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 195, loss 0.9634070251516533, train accuracy: 99.78%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 196, loss 0.9795732269737911, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 197, loss 1.2565338616896422, train accuracy: 99.56%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 198, loss 1.0216701323326807, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 199, loss 1.1273157354326333, train accuracy: 99.56%
Validation accuracy: 67.00%
Best Valid accuracy: 75.00%
Epoch 200, loss 0.9398573362633518, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 201, loss 1.0130739579698966, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 202, loss 0.9150161640238796, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 203, loss 1.2203564087146093, train accuracy: 99.56%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 204, loss 0.9756937771330216, train accuracy: 99.78%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 205, loss 1.0623148691766346, train accuracy: 99.56%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 206, loss 0.986333989000686, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 207, loss 0.9145328198758849, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 208, loss 0.948967628306435, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 209, loss 0.9032597158229005, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 210, loss 0.9450682024975949, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 211, loss 1.0332712807494264, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 212, loss 0.9675592246168382, train accuracy: 99.78%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 213, loss 0.9837480204459473, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 214, loss 1.0606555887104405, train accuracy: 99.56%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 215, loss 0.8621250519318845, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 216, loss 1.0687305121279482, train accuracy: 99.33%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 217, loss 0.8135977704158955, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 218, loss 0.7756284894509904, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 219, loss 0.8420133837803417, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 220, loss 0.8644477112705214, train accuracy: 99.78%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 221, loss 0.9116501545612147, train accuracy: 99.78%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 222, loss 0.8308872951560958, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 223, loss 0.8713668997433014, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 224, loss 0.8426389576324846, train accuracy: 99.78%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 225, loss 0.8985657502891257, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 226, loss 0.9604698506613207, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 227, loss 0.8915139155703375, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 228, loss 0.8524013747405459, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 229, loss 0.9557923248545532, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 230, loss 0.9234718552195147, train accuracy: 99.78%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 231, loss 0.9150617760862348, train accuracy: 99.78%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 232, loss 0.7734523528548728, train accuracy: 99.78%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 233, loss 0.8333869456279894, train accuracy: 99.78%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 234, loss 0.9030796357554185, train accuracy: 99.78%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 235, loss 0.8655848126871939, train accuracy: 99.56%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 236, loss 0.8224821613807943, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 237, loss 0.911125743275196, train accuracy: 99.78%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 238, loss 0.8968180624313714, train accuracy: 99.56%
Validation accuracy: 67.00%
Best Valid accuracy: 75.00%
Epoch 239, loss 0.8995428117993561, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 240, loss 0.8096581270973658, train accuracy: 99.78%
Validation accuracy: 67.00%
Best Valid accuracy: 75.00%
Epoch 241, loss 0.770490561072083, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 242, loss 0.7611170018636129, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 243, loss 0.8613534730032119, train accuracy: 99.78%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 244, loss 0.8463739553255861, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 245, loss 0.7546618447289194, train accuracy: 99.78%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 246, loss 0.869395126586303, train accuracy: 99.56%
Validation accuracy: 68.00%
Best Valid accuracy: 75.00%
Epoch 247, loss 0.7437010834690522, train accuracy: 99.78%
Validation accuracy: 69.00%
Best Valid accuracy: 75.00%
Epoch 248, loss 0.9418174662839133, train accuracy: 99.78%
Validation accuracy: 67.00%
Best Valid accuracy: 75.00%
Epoch 249, loss 0.8003538997187754, train accuracy: 99.56%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 250, loss 0.6978996435029584, train accuracy: 99.78%
Validation accuracy: 66.00%
Best Valid accuracy: 75.00%

real    20m8,866s
user    45m26,666s
sys     1m51,736s
```

## `run_mnist_multiclass.py`
### maxpool2d
```
Epoch 1 loss 2.3149765015497734 valid acc 0/16
Epoch 1 loss 11.534672373718987 valid acc 0/16
Epoch 1 loss 11.55263145778749 valid acc 0/16
Epoch 1 loss 11.508383553124748 valid acc 1/16
Epoch 1 loss 11.674613573390179 valid acc 2/16
Epoch 1 loss 11.173135842335338 valid acc 6/16
Epoch 1 loss 11.346631510907255 valid acc 5/16
Epoch 1 loss 11.456201140303232 valid acc 7/16
Epoch 1 loss 11.311197087543826 valid acc 7/16
Epoch 1 loss 11.377846755625633 valid acc 6/16
Epoch 1 loss 11.291132953366365 valid acc 7/16
Epoch 1 loss 11.012243918717969 valid acc 7/16
Epoch 1 loss 10.716160314493724 valid acc 7/16
Epoch 1 loss 10.001706797886811 valid acc 8/16
Epoch 1 loss 10.300462708991216 valid acc 8/16
Epoch 1 loss 9.349084013812428 valid acc 5/16
Epoch 1 loss 10.449917216865227 valid acc 5/16
Epoch 1 loss 10.22390221284442 valid acc 8/16
Epoch 1 loss 9.86312007270566 valid acc 11/16
Epoch 1 loss 9.636237938060324 valid acc 11/16
Epoch 1 loss 8.572738561166304 valid acc 11/16
Epoch 1 loss 6.886706349192287 valid acc 9/16
Epoch 1 loss 6.1837283239338205 valid acc 8/16
Epoch 1 loss 6.318266322265792 valid acc 8/16
Epoch 1 loss 6.152107078774194 valid acc 11/16
Epoch 1 loss 6.756301180146947 valid acc 9/16
Epoch 1 loss 6.335648967268453 valid acc 12/16
Epoch 1 loss 5.941873129103772 valid acc 11/16
Epoch 1 loss 5.1434332748163465 valid acc 10/16
Epoch 1 loss 4.7610013321404825 valid acc 12/16
Epoch 1 loss 6.039822985436706 valid acc 11/16
Epoch 1 loss 4.669388149382354 valid acc 11/16
Epoch 1 loss 4.432702898328159 valid acc 12/16
Epoch 1 loss 5.701185796928723 valid acc 13/16
Epoch 1 loss 5.303467991921767 valid acc 11/16
Epoch 1 loss 4.878095367830085 valid acc 10/16
Epoch 1 loss 4.429626351806192 valid acc 10/16
Epoch 1 loss 5.088956612457945 valid acc 14/16
Epoch 1 loss 3.734705929671257 valid acc 11/16
Epoch 1 loss 4.655403160930244 valid acc 12/16
Epoch 1 loss 4.319150465678254 valid acc 13/16
Epoch 1 loss 4.2551097107076465 valid acc 11/16
Epoch 1 loss 5.284060391936274 valid acc 13/16
Epoch 1 loss 4.398518592902377 valid acc 10/16
Epoch 1 loss 5.558639885381567 valid acc 13/16
Epoch 1 loss 4.0050268104388955 valid acc 13/16
Epoch 1 loss 4.658826850311506 valid acc 13/16
Epoch 1 loss 4.390177768593043 valid acc 14/16
Epoch 1 loss 3.7699162661296275 valid acc 15/16
Epoch 1 loss 3.959045067249553 valid acc 14/16
Epoch 1 loss 2.8670060619856383 valid acc 12/16
Epoch 1 loss 4.041964066697734 valid acc 12/16
Epoch 1 loss 3.938157554120327 valid acc 13/16
Epoch 1 loss 3.193404975805 valid acc 11/16
Epoch 1 loss 4.840730700805508 valid acc 13/16
Epoch 1 loss 4.149131057642904 valid acc 14/16
Epoch 1 loss 3.937500049586254 valid acc 12/16
Epoch 1 loss 4.150662041988793 valid acc 13/16
Epoch 1 loss 3.6908240019343603 valid acc 13/16
Epoch 1 loss 3.89321536570914 valid acc 13/16
Epoch 1 loss 3.216414770304893 valid acc 14/16
Epoch 1 loss 3.563193805761405 valid acc 14/16
Epoch 1 loss 3.554857532637138 valid acc 12/16
Epoch 2 loss 0.5762629886216923 valid acc 12/16
Epoch 2 loss 5.454321287811179 valid acc 12/16
Epoch 2 loss 3.6502716820183583 valid acc 13/16
Epoch 2 loss 4.368824181001754 valid acc 13/16
Epoch 2 loss 3.1349964291484627 valid acc 11/16
Epoch 2 loss 4.014217826234886 valid acc 14/16
Epoch 2 loss 2.6428524799739472 valid acc 12/16
Epoch 2 loss 3.379043965244014 valid acc 12/16
Epoch 2 loss 3.3778810006189652 valid acc 13/16
Epoch 2 loss 1.8115544817576303 valid acc 12/16
Epoch 2 loss 3.4199555264924077 valid acc 14/16
Epoch 2 loss 4.249146276577329 valid acc 14/16
Epoch 2 loss 3.1150323492586844 valid acc 14/16
Epoch 2 loss 3.8704846188923545 valid acc 12/16
Epoch 2 loss 3.9445533694572776 valid acc 13/16
Epoch 2 loss 3.1539756251250903 valid acc 15/16
Epoch 2 loss 5.323109633664386 valid acc 13/16
Epoch 2 loss 2.338555346541887 valid acc 14/16
Epoch 2 loss 2.7292492216722213 valid acc 13/16
Epoch 2 loss 2.496694081269553 valid acc 14/16
Epoch 2 loss 2.033599638663972 valid acc 14/16
Epoch 2 loss 1.8611856341588742 valid acc 14/16
Epoch 2 loss 1.3097910543548106 valid acc 15/16
Epoch 2 loss 1.619054616588447 valid acc 13/16
Epoch 2 loss 2.473156428286767 valid acc 14/16
Epoch 2 loss 2.132836808500598 valid acc 14/16
Epoch 2 loss 1.9076250500620515 valid acc 13/16
Epoch 2 loss 2.6240745897387576 valid acc 15/16
Epoch 2 loss 2.0536520440539645 valid acc 16/16
Epoch 2 loss 1.4556179074455087 valid acc 15/16
Epoch 2 loss 2.2616940873623577 valid acc 13/16
Epoch 2 loss 2.552596437770168 valid acc 13/16
Epoch 2 loss 1.9472648172689122 valid acc 13/16
Epoch 2 loss 2.445036606750428 valid acc 16/16
Epoch 2 loss 2.4831581467349273 valid acc 15/16
Epoch 2 loss 2.1298165690784288 valid acc 14/16
Epoch 2 loss 2.0895387484945496 valid acc 13/16
Epoch 2 loss 2.7157603596304307 valid acc 14/16
Epoch 2 loss 2.298829557105336 valid acc 15/16
Epoch 2 loss 2.7378480469573576 valid acc 15/16
Epoch 2 loss 2.1997254238970836 valid acc 12/16
Epoch 2 loss 2.756898894251998 valid acc 16/16
Epoch 2 loss 2.0991944772004145 valid acc 15/16
Epoch 2 loss 1.7435641828871855 valid acc 16/16
Epoch 2 loss 2.750354677746159 valid acc 16/16
Epoch 2 loss 2.614546800874299 valid acc 15/16
Epoch 2 loss 1.9684369546041802 valid acc 15/16
Epoch 2 loss 3.959229563235838 valid acc 16/16
Epoch 2 loss 1.5466582646489624 valid acc 15/16
Epoch 2 loss 1.4588834240263164 valid acc 16/16
Epoch 2 loss 2.3923996469621036 valid acc 13/16
Epoch 2 loss 3.5172472237477246 valid acc 14/16
Epoch 2 loss 3.197358066649933 valid acc 13/16
Epoch 2 loss 1.6355658175364705 valid acc 13/16
Epoch 2 loss 3.739563923471632 valid acc 12/16
Epoch 2 loss 1.9474145245419416 valid acc 12/16
Epoch 2 loss 1.870455257948186 valid acc 13/16
Epoch 2 loss 2.1651714897722067 valid acc 13/16
Epoch 2 loss 2.257041383840161 valid acc 13/16
Epoch 2 loss 2.6220134906990338 valid acc 13/16
Epoch 2 loss 1.915645077838338 valid acc 14/16
Epoch 2 loss 2.5559962220777575 valid acc 14/16
Epoch 2 loss 2.4624660261318567 valid acc 14/16
Epoch 3 loss 0.31283698308917784 valid acc 13/16
Epoch 3 loss 2.5427580857837766 valid acc 13/16
Epoch 3 loss 1.6828325979232293 valid acc 14/16
Epoch 3 loss 2.464504309750894 valid acc 12/16
Epoch 3 loss 1.3389126531014122 valid acc 12/16
Epoch 3 loss 1.5569553264924028 valid acc 14/16
Epoch 3 loss 2.684016774871975 valid acc 11/16
Epoch 3 loss 2.8138769140806943 valid acc 14/16
Epoch 3 loss 2.885280828703745 valid acc 15/16
Epoch 3 loss 1.6933340491794666 valid acc 15/16
Epoch 3 loss 1.9155460510856146 valid acc 15/16
Epoch 3 loss 2.3213532410568045 valid acc 14/16
Epoch 3 loss 2.742997288615806 valid acc 13/16
Epoch 3 loss 3.01055037817007 valid acc 13/16
Epoch 3 loss 2.118965179224685 valid acc 14/16
Epoch 3 loss 2.761225707201663 valid acc 14/16
Epoch 3 loss 3.0292689037609426 valid acc 15/16
Epoch 3 loss 2.2223815217269594 valid acc 14/16
Epoch 3 loss 2.80162791612728 valid acc 14/16
Epoch 3 loss 2.539735035420096 valid acc 15/16
Epoch 3 loss 1.7244912793571183 valid acc 13/16
Epoch 3 loss 1.3649046065597439 valid acc 12/16
Epoch 3 loss 0.8290168354809961 valid acc 15/16
Epoch 3 loss 1.3661580317921382 valid acc 15/16
Epoch 3 loss 1.6277791357336355 valid acc 14/16
Epoch 3 loss 1.66250742942319 valid acc 14/16
Epoch 3 loss 1.2893307428236267 valid acc 14/16
Epoch 3 loss 1.7346159016242026 valid acc 16/16
Epoch 3 loss 1.3868109608695403 valid acc 16/16
Epoch 3 loss 0.7389781302036067 valid acc 16/16
Epoch 3 loss 1.3936986597419885 valid acc 14/16
Epoch 3 loss 3.017040640670353 valid acc 14/16
Epoch 3 loss 1.5824428190076474 valid acc 14/16
Epoch 3 loss 3.803488324584057 valid acc 13/16
Epoch 3 loss 3.8943119632249843 valid acc 15/16
Epoch 3 loss 2.266717333812964 valid acc 14/16
Epoch 3 loss 2.556779932717191 valid acc 14/16
Epoch 3 loss 1.5061748255816734 valid acc 14/16
Epoch 3 loss 2.5576500132880042 valid acc 13/16
Epoch 3 loss 2.1676049434358595 valid acc 15/16
Epoch 3 loss 3.528531512428276 valid acc 12/16
Epoch 3 loss 18.143271222756777 valid acc 11/16
Epoch 3 loss 2.915227331619021 valid acc 13/16
Epoch 3 loss 2.501294525214558 valid acc 12/16
Epoch 3 loss 2.0776723313263554 valid acc 14/16
Epoch 3 loss 2.3566458186291506 valid acc 15/16
Epoch 3 loss 2.08598112992458 valid acc 14/16
Epoch 3 loss 3.05334048671918 valid acc 14/16
Epoch 3 loss 2.0470567781895586 valid acc 14/16
Epoch 3 loss 1.8798705651540712 valid acc 13/16
Epoch 3 loss 1.4889060005467774 valid acc 14/16
Epoch 3 loss 1.7746330477259478 valid acc 14/16
Epoch 3 loss 2.702552066731676 valid acc 15/16
Epoch 3 loss 1.4864995779164973 valid acc 14/16
Epoch 3 loss 6.577156918420711 valid acc 6/16
Epoch 3 loss 16.31938566395581 valid acc 13/16
Epoch 3 loss 2.546655112695973 valid acc 13/16
Epoch 3 loss 2.7089342336381965 valid acc 14/16
Epoch 3 loss 2.4002382599726104 valid acc 13/16
Epoch 3 loss 2.666845620644146 valid acc 14/16
Epoch 3 loss 2.052100967691217 valid acc 13/16
Epoch 3 loss 1.8794962118111824 valid acc 14/16
Epoch 3 loss 2.377380868946885 valid acc 13/16
Epoch 4 loss 0.28650375470965495 valid acc 13/16
Epoch 4 loss 2.6356390580899065 valid acc 14/16
Epoch 4 loss 1.7692073201275256 valid acc 14/16
Epoch 4 loss 2.0695168691903936 valid acc 14/16
Epoch 4 loss 2.6744872997121054 valid acc 13/16
Epoch 4 loss 2.0031593899871996 valid acc 14/16
Epoch 4 loss 2.570089980088043 valid acc 13/16
Epoch 4 loss 3.430958655008013 valid acc 11/16
Epoch 4 loss 3.0032296811385626 valid acc 14/16
Epoch 4 loss 1.8307752754158846 valid acc 15/16
Epoch 4 loss 2.42887240483995 valid acc 16/16
Epoch 4 loss 3.4932704413137623 valid acc 13/16
Epoch 4 loss 3.630884667067246 valid acc 13/16
Epoch 4 loss 2.9793264091533405 valid acc 14/16
Epoch 4 loss 2.791950197226563 valid acc 14/16
Epoch 4 loss 2.198381444819521 valid acc 15/16
Epoch 4 loss 2.9854248596551223 valid acc 16/16
Epoch 4 loss 3.278889012947885 valid acc 14/16
Epoch 4 loss 2.4344290582456174 valid acc 14/16
Epoch 4 loss 3.1837320725125156 valid acc 14/16
Epoch 4 loss 3.022964405812476 valid acc 14/16
Epoch 4 loss 1.4362770734733155 valid acc 16/16
Epoch 4 loss 0.8128679663478319 valid acc 15/16
Epoch 4 loss 1.1750699324402454 valid acc 16/16
Epoch 4 loss 1.5311012640156036 valid acc 15/16
Epoch 4 loss 1.8299944589011996 valid acc 15/16
Epoch 4 loss 1.6852414841273973 valid acc 16/16
Epoch 4 loss 1.9121515465922332 valid acc 16/16
Epoch 4 loss 1.035753586500583 valid acc 15/16
Epoch 4 loss 0.7305046544963564 valid acc 16/16
Epoch 4 loss 1.8790751290684422 valid acc 16/16
Epoch 4 loss 3.1979655610867805 valid acc 14/16
Epoch 4 loss 1.0124690148039193 valid acc 15/16
Epoch 4 loss 1.4238980678619813 valid acc 16/16
Epoch 4 loss 2.16435316948864 valid acc 16/16
Epoch 4 loss 2.207399806038705 valid acc 15/16
Epoch 4 loss 1.8416694910913627 valid acc 15/16
Epoch 4 loss 1.4532232512556806 valid acc 14/16
Epoch 4 loss 2.1362897714628453 valid acc 14/16
Epoch 4 loss 2.566432564581273 valid acc 14/16
Epoch 4 loss 0.7925259526750651 valid acc 15/16
Epoch 4 loss 2.939714575484328 valid acc 16/16
Epoch 4 loss 1.8541734024842411 valid acc 14/16
Epoch 4 loss 2.4361461865491787 valid acc 13/16
Epoch 4 loss 1.682189331674322 valid acc 15/16
Epoch 4 loss 2.934286009023305 valid acc 14/16
Epoch 4 loss 3.1712414478413047 valid acc 13/16
Epoch 4 loss 2.0235749509692504 valid acc 15/16
Epoch 4 loss 2.1596707946160505 valid acc 15/16
Epoch 4 loss 2.226638544795475 valid acc 15/16
Epoch 4 loss 1.5835569317808575 valid acc 15/16
Epoch 4 loss 3.7137195735887896 valid acc 13/16
Epoch 4 loss 6.310305658290938 valid acc 14/16
Epoch 4 loss 1.50289115057218 valid acc 14/16
Epoch 4 loss 5.140481624279958 valid acc 14/16
Epoch 4 loss 3.3257506106385426 valid acc 13/16
Epoch 4 loss 6.6694095688581445 valid acc 12/16
Epoch 4 loss 3.2206604565513195 valid acc 13/16
Epoch 4 loss 3.1526037279192423 valid acc 12/16
Epoch 4 loss 5.336825942290144 valid acc 13/16
Epoch 4 loss 2.159810773334192 valid acc 14/16
Epoch 4 loss 3.8589424252792037 valid acc 15/16
Epoch 4 loss 2.6241469235449415 valid acc 12/16
Epoch 5 loss 0.044927332001445996 valid acc 13/16
Epoch 5 loss 3.0882524566357294 valid acc 14/16
Epoch 5 loss 3.011764029205601 valid acc 12/16
Epoch 5 loss 3.2580483883285627 valid acc 14/16
Epoch 5 loss 4.357624284455806 valid acc 13/16
Epoch 5 loss 2.901635078791818 valid acc 13/16
Epoch 5 loss 6.928972904906226 valid acc 11/16
Epoch 5 loss 4.312128805715406 valid acc 10/16
Epoch 5 loss 8.46373910186301 valid acc 10/16
Epoch 5 loss 4.409923039103434 valid acc 14/16
Epoch 5 loss 3.818531663781303 valid acc 15/16
Epoch 5 loss 2.7217830081533254 valid acc 14/16
Epoch 5 loss 3.5696743110438276 valid acc 15/16
Epoch 5 loss 4.691859538607356 valid acc 15/16
Epoch 5 loss 3.754496394979223 valid acc 13/16
Epoch 5 loss 4.413848123981695 valid acc 15/16
Epoch 5 loss 3.104005886667746 valid acc 14/16
Epoch 5 loss 2.8416070840454304 valid acc 15/16
Epoch 5 loss 2.7540605844144945 valid acc 14/16
Epoch 5 loss 2.072976176262526 valid acc 14/16
Epoch 5 loss 1.6089497557413448 valid acc 14/16
Epoch 5 loss 1.654782897440818 valid acc 14/16
Epoch 5 loss 0.8573637472377234 valid acc 13/16
Epoch 5 loss 1.5262956159403283 valid acc 14/16
Epoch 5 loss 2.520751497459898 valid acc 15/16
Epoch 5 loss 2.355404379117264 valid acc 15/16
Epoch 5 loss 2.8966802898181983 valid acc 15/16
Epoch 5 loss 2.4419021394910905 valid acc 14/16
Epoch 5 loss 2.2935568724894626 valid acc 13/16
Epoch 5 loss 1.5604368637368555 valid acc 15/16
Epoch 5 loss 2.468152826498411 valid acc 13/16
Epoch 5 loss 2.479032625103166 valid acc 14/16
Epoch 5 loss 1.5675038827495122 valid acc 13/16
Epoch 5 loss 1.927701928680786 valid acc 14/16
Epoch 5 loss 2.5189194458964037 valid acc 13/16
Epoch 5 loss 2.382218586370587 valid acc 13/16
Epoch 5 loss 2.785518982081795 valid acc 12/16
Epoch 5 loss 1.4524693193706881 valid acc 14/16
Epoch 5 loss 2.426951733493846 valid acc 13/16
Epoch 5 loss 2.3861624709542166 valid acc 13/16
Epoch 5 loss 2.037478322030883 valid acc 12/16
Epoch 5 loss 5.210401213497664 valid acc 14/16
Epoch 5 loss 1.613039520288805 valid acc 14/16
Epoch 5 loss 1.644385495687846 valid acc 13/16
Epoch 5 loss 3.2947984258485543 valid acc 16/16
Epoch 5 loss 1.8811945805809869 valid acc 15/16
Epoch 5 loss 2.288386511464828 valid acc 16/16
Epoch 5 loss 3.6552590708981874 valid acc 15/16
Epoch 5 loss 1.8294663756664007 valid acc 15/16
Epoch 5 loss 1.2846802721695898 valid acc 15/16
Epoch 5 loss 1.6710774335549845 valid acc 15/16
Epoch 5 loss 1.3431842444003175 valid acc 15/16
Epoch 5 loss 2.4239221209774087 valid acc 14/16
Epoch 5 loss 2.2496060776958546 valid acc 13/16
Epoch 5 loss 2.4773614843478757 valid acc 13/16
Epoch 5 loss 1.1890416187464976 valid acc 13/16
Epoch 5 loss 1.8767527169836247 valid acc 15/16
Epoch 5 loss 2.0275341607019124 valid acc 13/16
Epoch 5 loss 2.9811663215820947 valid acc 10/16
Epoch 5 loss 5.867799792424507 valid acc 13/16
Epoch 5 loss 3.1002267217251807 valid acc 12/16
Epoch 5 loss 2.3344815960424223 valid acc 13/16
Epoch 5 loss 2.7178584629832123 valid acc 12/16
Epoch 6 loss 0.3213581050605552 valid acc 13/16
Epoch 6 loss 3.2447227045594413 valid acc 12/16
Epoch 6 loss 1.9954273557439344 valid acc 12/16
Epoch 6 loss 2.272932276930949 valid acc 12/16
Epoch 6 loss 1.4231837591356449 valid acc 12/16
Epoch 6 loss 1.7188382087625287 valid acc 14/16
Epoch 6 loss 1.66627138391474 valid acc 13/16
Epoch 6 loss 2.9687183895361313 valid acc 13/16
Epoch 6 loss 2.6976140687319248 valid acc 14/16
Epoch 6 loss 1.3052114415976923 valid acc 14/16
Epoch 6 loss 1.1205199810140436 valid acc 15/16
Epoch 6 loss 2.32754528745754 valid acc 14/16
Epoch 6 loss 2.4234907923810836 valid acc 15/16
Epoch 6 loss 6.124318056593167 valid acc 12/16
Epoch 6 loss 16.32561898123009 valid acc 8/16
Epoch 6 loss 29.284192473715986 valid acc 7/16
Epoch 6 loss 50.84914953940939 valid acc 5/16
Epoch 6 loss 53.23097806807907 valid acc 8/16
Epoch 6 loss 7.605329887191203 valid acc 6/16
Epoch 6 loss 10.434467024243661 valid acc 5/16
Epoch 6 loss 19.969374051986833 valid acc 6/16
Epoch 6 loss 16.53844353232468 valid acc 3/16
Epoch 6 loss 11.357396276564502 valid acc 2/16
Epoch 6 loss 11.826032922907102 valid acc 2/16
Epoch 6 loss 11.935467628279488 valid acc 2/16
Epoch 6 loss 17.399111252013395 valid acc 2/16
Epoch 6 loss 13.764531841608443 valid acc 2/16
Epoch 6 loss 11.40484152796273 valid acc 2/16
Epoch 6 loss 11.463136674791478 valid acc 2/16
Epoch 6 loss 11.512390246213915 valid acc 1/16
Epoch 6 loss 11.60749067195444 valid acc 2/16
Epoch 6 loss 11.490196238853057 valid acc 1/16
Epoch 6 loss 38.87692757870498 valid acc 1/16
Epoch 6 loss 11.523618344252956 valid acc 2/16
Epoch 6 loss 11.505363288395593 valid acc 2/16
Epoch 6 loss 11.508738817095978 valid acc 2/16
Epoch 6 loss 11.504558795479879 valid acc 2/16
Epoch 6 loss 11.539140781759478 valid acc 2/16
Epoch 6 loss 11.508347215314325 valid acc 2/16
Epoch 6 loss 11.466603058804436 valid acc 2/16
Epoch 6 loss 11.523568329633349 valid acc 2/16
Epoch 6 loss 11.515233487339973 valid acc 2/16
Epoch 6 loss 11.494765551964864 valid acc 1/16
Epoch 6 loss 11.52786776816361 valid acc 2/16
Epoch 6 loss 11.500133890677649 valid acc 2/16
Epoch 6 loss 11.509455507975249 valid acc 2/16
Epoch 6 loss 11.532987283606479 valid acc 2/16
Epoch 6 loss 16.672014337884253 valid acc 2/16
Epoch 6 loss 11.522178796236563 valid acc 2/16
Epoch 6 loss 11.510496133495515 valid acc 2/16
Epoch 6 loss 11.494033021966994 valid acc 2/16
Epoch 6 loss 11.50029188784772 valid acc 2/16
Epoch 6 loss 11.463688921978044 valid acc 0/16
Epoch 6 loss 17.3826936258709 valid acc 3/16
Epoch 6 loss 11.258435113326195 valid acc 3/16
Epoch 6 loss 11.258701699205204 valid acc 3/16
Epoch 6 loss 11.30635836795334 valid acc 2/16
Epoch 6 loss 11.631700524968705 valid acc 2/16
Epoch 6 loss 11.528009869882629 valid acc 2/16
Epoch 6 loss 11.511094040573184 valid acc 2/16
Epoch 6 loss 11.490201571161949 valid acc 2/16
Epoch 6 loss 11.517339609496776 valid acc 2/16
Epoch 6 loss 11.454784502064392 valid acc 3/16
Epoch 7 loss 2.7940865937694017 valid acc 2/16
Epoch 7 loss 11.496467317362265 valid acc 2/16
Epoch 7 loss 11.565866123507181 valid acc 2/16
Epoch 7 loss 11.53229937117379 valid acc 2/16
Epoch 7 loss 11.51972971519243 valid acc 2/16
Epoch 7 loss 11.452635857566534 valid acc 2/16
Epoch 7 loss 11.511453541464892 valid acc 2/16
Epoch 7 loss 11.527561251406151 valid acc 2/16
Epoch 7 loss 11.515756736653532 valid acc 2/16
Epoch 7 loss 11.508509104374872 valid acc 2/16
Epoch 7 loss 11.511771544160252 valid acc 2/16
Epoch 7 loss 11.524487399764581 valid acc 2/16
Epoch 7 loss 11.53027103222002 valid acc 2/16
Epoch 7 loss 11.499178132248147 valid acc 2/16
Epoch 7 loss 11.496025559111219 valid acc 2/16
Epoch 7 loss 11.379821063704014 valid acc 2/16
Epoch 7 loss 11.3748877077538 valid acc 3/16
Epoch 7 loss 11.551843956752126 valid acc 2/16
Epoch 7 loss 11.520163058147284 valid acc 2/16
Epoch 7 loss 11.387613541124233 valid acc 2/16
Epoch 7 loss 11.514822457482687 valid acc 2/16
Epoch 7 loss 11.493235054396582 valid acc 2/16
Epoch 7 loss 11.418180165257972 valid acc 3/16
Epoch 7 loss 12.467704339936969 valid acc 2/16
Epoch 7 loss 11.5231887800725 valid acc 2/16
Epoch 7 loss 11.52066362989073 valid acc 2/16
Epoch 7 loss 11.518500981095094 valid acc 2/16
Epoch 7 loss 11.505677471769543 valid acc 2/16
Epoch 7 loss 11.503620490791148 valid acc 2/16
Epoch 7 loss 11.524338224324804 valid acc 2/16
Epoch 7 loss 11.507086401918546 valid acc 2/16
Epoch 7 loss 11.474085349166224 valid acc 2/16
Epoch 7 loss 11.541237571583022 valid acc 2/16
Epoch 7 loss 11.521259099692934 valid acc 2/16
Epoch 7 loss 11.49412732392304 valid acc 2/16
Epoch 7 loss 11.499926014840373 valid acc 2/16
Epoch 7 loss 11.486867299690822 valid acc 2/16
Epoch 7 loss 11.545516099167408 valid acc 2/16
Epoch 7 loss 11.510586861730983 valid acc 2/16
Epoch 7 loss 11.464337242763934 valid acc 2/16
Epoch 7 loss 11.529942887631616 valid acc 2/16
Epoch 7 loss 11.505171035334403 valid acc 2/16
Epoch 7 loss 11.492049935376748 valid acc 2/16
Epoch 7 loss 11.517111078647746 valid acc 2/16
Epoch 7 loss 11.498780505488385 valid acc 2/16
Epoch 7 loss 11.506350981893231 valid acc 2/16
Epoch 7 loss 11.548538112183568 valid acc 2/16
Epoch 7 loss 11.474202546698642 valid acc 2/16
Epoch 7 loss 11.507359841725672 valid acc 2/16
Epoch 7 loss 11.511678327551682 valid acc 2/16
Epoch 7 loss 11.492024706637705 valid acc 2/16
Epoch 7 loss 11.506363837609474 valid acc 2/16
Epoch 7 loss 11.476642856753106 valid acc 2/16
Epoch 7 loss 11.5172559639392 valid acc 2/16
Epoch 7 loss 11.535406504331185 valid acc 2/16
Epoch 7 loss 11.503839932657902 valid acc 2/16
Epoch 7 loss 11.49585306907899 valid acc 2/16
Epoch 7 loss 11.49467060293702 valid acc 2/16
Epoch 7 loss 11.535575505271872 valid acc 2/16
Epoch 7 loss 11.52168744484943 valid acc 2/16
Epoch 7 loss 11.470669407555995 valid acc 2/16
Epoch 7 loss 11.513870638724157 valid acc 2/16
Epoch 7 loss 11.479813754281373 valid acc 2/16
Epoch 8 loss 2.2954245299317897 valid acc 2/16
Epoch 8 loss 11.48849211513198 valid acc 2/16
Epoch 8 loss 11.486960799523601 valid acc 2/16
Epoch 8 loss 11.523088266395742 valid acc 2/16
Epoch 8 loss 11.524123151670336 valid acc 2/16
Epoch 8 loss 11.441675076264344 valid acc 2/16
Epoch 8 loss 11.504706288746894 valid acc 2/16
Epoch 8 loss 11.531745665871902 valid acc 2/16
Epoch 8 loss 11.505873791397587 valid acc 2/16
Epoch 8 loss 11.508058178868422 valid acc 2/16
Epoch 8 loss 11.527260967899302 valid acc 2/16
Epoch 8 loss 11.514331712273407 valid acc 2/16
Epoch 8 loss 11.515731336981275 valid acc 2/16
Epoch 8 loss 11.482742148499256 valid acc 2/16
Epoch 8 loss 11.487871127990616 valid acc 2/16
Epoch 8 loss 11.465956676232258 valid acc 2/16
Epoch 8 loss 11.493978978336783 valid acc 2/16
Epoch 8 loss 11.541285657545775 valid acc 2/16
Epoch 8 loss 11.520094045317382 valid acc 2/16
Epoch 8 loss 11.526630490132725 valid acc 2/16
Epoch 8 loss 11.514146021804443 valid acc 2/16
Epoch 8 loss 11.484000238978524 valid acc 2/16
Epoch 8 loss 11.484321395553259 valid acc 2/16
Epoch 8 loss 11.509305421390259 valid acc 2/16
Epoch 8 loss 11.524366955855372 valid acc 2/16
Epoch 8 loss 11.52519220517636 valid acc 2/16
Epoch 8 loss 11.516637658770826 valid acc 2/16
Epoch 8 loss 11.497543847480808 valid acc 2/16
Epoch 8 loss 11.495474359920188 valid acc 2/16
Epoch 8 loss 11.536186408834519 valid acc 2/16
Epoch 8 loss 11.509623178625873 valid acc 2/16
Epoch 8 loss 11.460198540320617 valid acc 2/16
Epoch 8 loss 11.537836620418176 valid acc 2/16
Epoch 8 loss 11.521284009330792 valid acc 2/16
Epoch 8 loss 11.487356237431422 valid acc 2/16
Epoch 8 loss 11.49426317825668 valid acc 2/16
Epoch 8 loss 11.474565691807246 valid acc 2/16
Epoch 8 loss 11.552116186258981 valid acc 2/16
Epoch 8 loss 11.513469559215961 valid acc 2/16
Epoch 8 loss 11.463002590233279 valid acc 2/16
Epoch 8 loss 11.536031477004084 valid acc 2/16
Epoch 8 loss 11.496376083235171 valid acc 2/16
Epoch 8 loss 11.491094794882663 valid acc 2/16
Epoch 8 loss 11.510423172992052 valid acc 2/16
Epoch 8 loss 11.498327486805445 valid acc 2/16
Epoch 8 loss 11.50720064575384 valid acc 2/16
Epoch 8 loss 11.561008633658435 valid acc 2/16
Epoch 8 loss 11.46947763015317 valid acc 2/16
Epoch 8 loss 11.49644429108291 valid acc 2/16
Epoch 8 loss 11.513272548040076 valid acc 2/16
Epoch 8 loss 11.492220720599839 valid acc 2/16
Epoch 8 loss 11.511783991958941 valid acc 2/16
Epoch 8 loss 11.470787051962441 valid acc 2/16
Epoch 8 loss 11.506059088156404 valid acc 2/16
Epoch 8 loss 11.528537082896431 valid acc 2/16
Epoch 8 loss 11.496882541134637 valid acc 2/16
Epoch 8 loss 11.493090124740238 valid acc 2/16
Epoch 8 loss 11.480379348574548 valid acc 2/16
Epoch 8 loss 11.542324378071008 valid acc 2/16
Epoch 8 loss 11.531416063813715 valid acc 2/16
Epoch 8 loss 11.455609135805513 valid acc 2/16
Epoch 8 loss 11.511662777155266 valid acc 2/16
Epoch 8 loss 11.469969420510813 valid acc 2/16
Epoch 9 loss 2.2920787300206693 valid acc 2/16
Epoch 9 loss 11.483351551465333 valid acc 2/16
Epoch 9 loss 11.480406782422335 valid acc 2/16
Epoch 9 loss 11.518144045372905 valid acc 2/16
Epoch 9 loss 11.527707963246158 valid acc 2/16
Epoch 9 loss 11.43352863812283 valid acc 2/16
Epoch 9 loss 11.501381658477234 valid acc 2/16
Epoch 9 loss 11.536560434690806 valid acc 2/16
Epoch 9 loss 11.499447699546995 valid acc 2/16
Epoch 9 loss 11.508060852562654 valid acc 2/16
Epoch 9 loss 11.539615593262438 valid acc 2/16
Epoch 9 loss 11.50732660628051 valid acc 2/16
Epoch 9 loss 11.506025836178688 valid acc 2/16
Epoch 9 loss 11.471435389556548 valid acc 2/16
Epoch 9 loss 11.483015430940323 valid acc 2/16
Epoch 9 loss 11.456892441719692 valid acc 2/16
Epoch 9 loss 11.499375551814586 valid acc 2/16
Epoch 9 loss 11.536809776193858 valid acc 2/16
Epoch 9 loss 11.520773061659336 valid acc 2/16
Epoch 9 loss 11.530333443212141 valid acc 2/16
Epoch 9 loss 11.514078563985759 valid acc 2/16
Epoch 9 loss 11.47772711567281 valid acc 2/16
Epoch 9 loss 11.483664541751384 valid acc 2/16
Epoch 9 loss 11.508973177154344 valid acc 2/16
Epoch 9 loss 11.525785325686792 valid acc 2/16
Epoch 9 loss 11.529234811592495 valid acc 2/16
Epoch 9 loss 11.515953304378554 valid acc 2/16
Epoch 9 loss 11.492274673735043 valid acc 2/16
Epoch 9 loss 11.490164952895 valid acc 2/16
Epoch 9 loss 11.545620903179765 valid acc 2/16
Epoch 9 loss 11.512205585482189 valid acc 2/16
Epoch 9 loss 11.450660596663635 valid acc 2/16
Epoch 9 loss 11.535947682161197 valid acc 2/16
Epoch 9 loss 11.521953897586675 valid acc 2/16
Epoch 9 loss 11.483038886663484 valid acc 2/16
Epoch 9 loss 11.490703982867558 valid acc 2/16
Epoch 9 loss 11.466180406223847 valid acc 2/16
Epoch 9 loss 11.557624122628813 valid acc 2/16
Epoch 9 loss 11.516139220648858 valid acc 2/16
Epoch 9 loss 11.462615543671488 valid acc 2/16
Epoch 9 loss 11.541066075745377 valid acc 2/16
Epoch 9 loss 11.490545665742186 valid acc 2/16
Epoch 9 loss 11.490846564978659 valid acc 2/16
Epoch 9 loss 11.506138194962336 valid acc 2/16
Epoch 9 loss 11.498569594371824 valid acc 2/16
Epoch 9 loss 11.508399225764771 valid acc 2/16
Epoch 9 loss 11.57066413558505 valid acc 2/16
Epoch 9 loss 11.466514773963201 valid acc 2/16
Epoch 9 loss 11.4889364072211 valid acc 2/16
Epoch 9 loss 11.515023445669911 valid acc 2/16
Epoch 9 loss 11.492937525722821 valid acc 2/16
Epoch 9 loss 11.516243293380791 valid acc 2/16
Epoch 9 loss 11.467110137589522 valid acc 2/16
Epoch 9 loss 11.498297759028164 valid acc 2/16
Epoch 9 loss 11.524065726568777 valid acc 2/16
Epoch 9 loss 11.492240227726501 valid acc 2/16
Epoch 9 loss 11.491624529754262 valid acc 2/16
Epoch 9 loss 11.47041262608336 valid acc 2/16
Epoch 9 loss 11.547748551631095 valid acc 2/16
Epoch 9 loss 11.539017197585508 valid acc 2/16
Epoch 9 loss 11.445102828450072 valid acc 2/16
Epoch 9 loss 11.510446159441365 valid acc 2/16
Epoch 9 loss 11.463332661924827 valid acc 2/16
Epoch 10 loss 2.289749693658327 valid acc 2/16
Epoch 10 loss 11.479968240740638 valid acc 2/16
Epoch 10 loss 11.476080419910705 valid acc 2/16
Epoch 10 loss 11.514953680372273 valid acc 2/16
Epoch 10 loss 11.530807196235806 valid acc 2/16
Epoch 10 loss 11.428056176799103 valid acc 2/16
Epoch 10 loss 11.49938124669616 valid acc 2/16
Epoch 10 loss 11.540659223708087 valid acc 2/16
Epoch 10 loss 11.495241933816166 valid acc 2/16
Epoch 10 loss 11.508439453199749 valid acc 2/16
Epoch 10 loss 11.549115461054154 valid acc 2/16
Epoch 10 loss 11.502593879611032 valid acc 2/16
Epoch 10 loss 11.4994850152232 valid acc 2/16
Epoch 10 loss 11.463583863822215 valid acc 2/16
Epoch 10 loss 11.479965422343291 valid acc 2/16
Epoch 10 loss 11.450582079419839 valid acc 2/16
Epoch 10 loss 11.503776602514305 valid acc 2/16
Epoch 10 loss 11.533922807195445 valid acc 2/16
Epoch 10 loss 11.521656430963741 valid acc 2/16
Epoch 10 loss 11.533488979040538 valid acc 2/16
Epoch 10 loss 11.514307918675804 valid acc 2/16
Epoch 10 loss 11.473492365401377 valid acc 2/16
Epoch 10 loss 11.483474686739362 valid acc 2/16
Epoch 10 loss 11.509158604208103 valid acc 2/16
Epoch 10 loss 11.527058131800835 valid acc 2/16
Epoch 10 loss 11.532605354060578 valid acc 2/16
Epoch 10 loss 11.51582533683005 valid acc 2/16
Epoch 10 loss 11.488798659392366 valid acc 2/16
Epoch 10 loss 11.486609930763382 valid acc 2/16
Epoch 10 loss 11.552933236933384 valid acc 2/16
Epoch 10 loss 11.51451777282698 valid acc 2/16
Epoch 10 loss 11.444008807696614 valid acc 2/16
Epoch 10 loss 11.534862144001481 valid acc 2/16
Epoch 10 loss 11.522793659994157 valid acc 2/16
Epoch 10 loss 11.480239117772534 valid acc 2/16
Epoch 10 loss 11.488392361561907 valid acc 2/16
Epoch 10 loss 11.460356035156408 valid acc 2/16
Epoch 10 loss 11.562030946936757 valid acc 2/16
Epoch 10 loss 11.518390231584593 valid acc 2/16
Epoch 10 loss 11.462636477505784 valid acc 2/16
Epoch 10 loss 11.545073182000127 valid acc 2/16
Epoch 10 loss 11.486634936071653 valid acc 2/16
Epoch 10 loss 11.49085758025536 valid acc 2/16
Epoch 10 loss 11.503326814365337 valid acc 2/16
Epoch 10 loss 11.499078307510278 valid acc 2/16
Epoch 10 loss 11.50956715066747 valid acc 2/16
Epoch 10 loss 11.578014251199257 valid acc 2/16
Epoch 10 loss 11.464592871339894 valid acc 2/16
Epoch 10 loss 11.48369598880448 valid acc 2/16
Epoch 10 loss 11.516643479869037 valid acc 2/16
Epoch 10 loss 11.493771247055005 valid acc 2/16
Epoch 10 loss 11.519768926858898 valid acc 2/16
Epoch 10 loss 11.464755552405517 valid acc 2/16
Epoch 10 loss 11.492822171653685 valid acc 2/16
Epoch 10 loss 11.521125401089604 valid acc 2/16
Epoch 10 loss 11.489081228382583 valid acc 2/16
Epoch 10 loss 11.490859468321792 valid acc 2/16
Epoch 10 loss 11.463376905246266 valid acc 2/16
Epoch 10 loss 11.551971063364341 valid acc 2/16
Epoch 10 loss 11.54483750282742 valid acc 2/16
Epoch 10 loss 11.43769076118629 valid acc 2/16
Epoch 10 loss 11.50975614435627 valid acc 2/16
Epoch 10 loss 11.45880067083092 valid acc 2/16
Epoch 11 loss 2.2881100680440314 valid acc 2/16
Epoch 11 loss 11.477663661212173 valid acc 2/16
Epoch 11 loss 11.473164356344139 valid acc 2/16
Epoch 11 loss 11.512834218226871 valid acc 2/16
Epoch 11 loss 11.533328767289627 valid acc 2/16
Epoch 11 loss 11.424325442271979 valid acc 2/16
Epoch 11 loss 11.498139607123449 valid acc 2/16
Epoch 11 loss 11.544012522855432 valid acc 2/16
Epoch 11 loss 11.492456173562228 valid acc 2/16
Epoch 11 loss 11.508895655014726 valid acc 2/16
Epoch 11 loss 11.556328888803794 valid acc 2/16
Epoch 11 loss 11.499335473977045 valid acc 2/16
Epoch 11 loss 11.49506029993766 valid acc 2/16
Epoch 11 loss 11.458067091493136 valid acc 2/16
Epoch 11 loss 11.478030731018444 valid acc 2/16
Epoch 11 loss 11.44610739345589 valid acc 2/16
Epoch 11 loss 11.50725599266055 valid acc 2/16
Epoch 11 loss 11.532026625630325 valid acc 2/16
Epoch 11 loss 11.522511435447079 valid acc 2/16
Epoch 11 loss 11.536043531988351 valid acc 2/16
Epoch 11 loss 11.514583618513548 valid acc 2/16
Epoch 11 loss 11.470572545958426 valid acc 2/16
Epoch 11 loss 11.483459520202777 valid acc 2/16
Epoch 11 loss 11.509541094974272 valid acc 2/16
Epoch 11 loss 11.528068238798253 valid acc 2/16
Epoch 11 loss 11.535308508820265 valid acc 2/16
Epoch 11 loss 11.515937384837695 valid acc 2/16
Epoch 11 loss 11.486467375175407 valid acc 2/16
Epoch 11 loss 11.48417099777814 valid acc 2/16
Epoch 11 loss 11.558503317086828 valid acc 2/16
Epoch 11 loss 11.516459941345026 valid acc 2/16
Epoch 11 loss 11.4393106158956 valid acc 2/16
Epoch 11 loss 11.534208364979673 valid acc 2/16
Epoch 11 loss 11.52359502619197 valid acc 2/16
Epoch 11 loss 11.478394643423876 valid acc 2/16
Epoch 11 loss 11.486841824976857 valid acc 2/16
Epoch 11 loss 11.456245808001357 valid acc 2/16
Epoch 11 loss 11.56546727193022 valid acc 2/16
Epoch 11 loss 11.520194058717115 valid acc 2/16
Epoch 11 loss 11.462805334651826 valid acc 2/16
Epoch 11 loss 11.548187628315443 valid acc 2/16
Epoch 11 loss 11.483985934679602 valid acc 2/16
Epoch 11 loss 11.490932208187212 valid acc 2/16
Epoch 11 loss 11.501441083916443 valid acc 2/16
Epoch 11 loss 11.499645079931094 valid acc 2/16
Epoch 11 loss 11.510563639329652 valid acc 2/16
Epoch 11 loss 11.583546505744037 valid acc 2/16
Epoch 11 loss 11.463303426019346 valid acc 2/16
Epoch 11 loss 11.47999299478609 valid acc 2/16
Epoch 11 loss 11.518028842008787 valid acc 2/16
Epoch 11 loss 11.494547938263935 valid acc 2/16
Epoch 11 loss 11.52248794878136 valid acc 2/16
Epoch 11 loss 11.463219621307868 valid acc 2/16
Epoch 11 loss 11.488903340544226 valid acc 2/16
Epoch 11 loss 11.519175686745848 valid acc 2/16
Epoch 11 loss 11.486894353083699 valid acc 2/16
Epoch 11 loss 11.490470950403976 valid acc 2/16
Epoch 11 loss 11.458360977610827 valid acc 2/16
Epoch 11 loss 11.555192003368676 valid acc 2/16
Epoch 11 loss 11.549234743893265 valid acc 2/16
Epoch 11 loss 11.432414195268915 valid acc 2/16
Epoch 11 loss 11.509348126631734 valid acc 2/16
Epoch 11 loss 11.455672462801093 valid acc 2/16
Epoch 12 loss 2.2869453458657447 valid acc 2/16
Epoch 12 loss 11.4760448700873 valid acc 2/16
Epoch 12 loss 11.471162252877663 valid acc 2/16
Epoch 12 loss 11.511387494798408 valid acc 2/16
Epoch 12 loss 11.53530947060341 valid acc 2/16
Epoch 12 loss 11.421749845255418 valid acc 2/16
Epoch 12 loss 11.497342288756176 valid acc 2/16
Epoch 12 loss 11.546689774710977 valid acc 2/16
Epoch 12 loss 11.49059247922765 valid acc 2/16
Epoch 12 loss 11.509308596633172 valid acc 2/16
Epoch 12 loss 11.561759886489387 valid acc 2/16
Epoch 12 loss 11.497056061792954 valid acc 2/16
Epoch 12 loss 11.492060395604389 valid acc 2/16
Epoch 12 loss 11.454153229357896 valid acc 2/16
Epoch 12 loss 11.47679392882479 valid acc 2/16
Epoch 12 loss 11.442886973152287 valid acc 2/16
Epoch 12 loss 11.509953506215055 valid acc 2/16
Epoch 12 loss 11.530760634823546 valid acc 2/16
Epoch 12 loss 11.523249922299819 valid acc 2/16
Epoch 12 loss 11.538049746096988 valid acc 2/16
Epoch 12 loss 11.514812767045436 valid acc 2/16
Epoch 12 loss 11.468522952903959 valid acc 2/16
Epoch 12 loss 11.483490567890218 valid acc 2/16
Epoch 12 loss 11.509966011495653 valid acc 2/16
Epoch 12 loss 11.528814849628773 valid acc 2/16
Epoch 12 loss 11.537425263898873 valid acc 2/16
Epoch 12 loss 11.516133981017598 valid acc 2/16
Epoch 12 loss 11.484881169160083 valid acc 2/16
Epoch 12 loss 11.482462916908487 valid acc 2/16
Epoch 12 loss 11.562698215708714 valid acc 2/16
Epoch 12 loss 11.518032643578248 valid acc 2/16
Epoch 12 loss 11.435958210579734 valid acc 2/16
Epoch 12 loss 11.533791893596296 valid acc 2/16
Epoch 12 loss 11.524281124180977 valid acc 2/16
Epoch 12 loss 11.477162309379578 valid acc 2/16
Epoch 12 loss 11.485770823682076 valid acc 2/16
Epoch 12 loss 11.453307714249288 valid acc 2/16
Epoch 12 loss 11.568103279242113 valid acc 2/16
Epoch 12 loss 11.521595789295938 valid acc 2/16
Epoch 12 loss 11.46300408679915 valid acc 2/16
Epoch 12 loss 11.550571232258164 valid acc 2/16
Epoch 12 loss 11.482177282049337 valid acc 2/16
Epoch 12 loss 11.490995306764226 valid acc 2/16
Epoch 12 loss 11.500151449172932 valid acc 2/16
Epoch 12 loss 11.5001761620473 valid acc 2/16
Epoch 12 loss 11.511358574238113 valid acc 2/16
Epoch 12 loss 11.587678952500493 valid acc 2/16
Epoch 12 loss 11.462411034333215 valid acc 2/16
Epoch 12 loss 11.477349929567525 valid acc 2/16
Epoch 12 loss 11.519163039765344 valid acc 2/16
Epoch 12 loss 11.49520628361982 valid acc 2/16
Epoch 12 loss 11.524551053002863 valid acc 2/16
Epoch 12 loss 11.462201101164023 valid acc 2/16
Epoch 12 loss 11.486066832853297 valid acc 2/16
Epoch 12 loss 11.51787458281655 valid acc 2/16
Epoch 12 loss 11.485358360363493 valid acc 2/16
Epoch 12 loss 11.490283413228621 valid acc 2/16
Epoch 12 loss 11.454756805720395 valid acc 2/16
Epoch 12 loss 11.557616341874114 valid acc 2/16
Epoch 12 loss 11.552526704820213 valid acc 2/16
Epoch 12 loss 11.428631097407221 valid acc 2/16
Epoch 12 loss 11.50909373312954 valid acc 2/16
Epoch 12 loss 11.45349417053264 valid acc 2/16
Epoch 13 loss 2.2861121692850843 valid acc 2/16
Epoch 13 loss 11.474878371271956 valid acc 2/16
Epoch 13 loss 11.469765967992746 valid acc 2/16
Epoch 13 loss 11.510376259624708 valid acc 2/16
Epoch 13 loss 11.53683183824404 valid acc 2/16
Epoch 13 loss 11.419953021103588 valid acc 2/16
Epoch 13 loss 11.496812403668752 valid acc 2/16
Epoch 13 loss 11.548793861134586 valid acc 2/16
Epoch 13 loss 11.48933571059039 valid acc 2/16
Epoch 13 loss 11.509640793612913 valid acc 2/16
Epoch 13 loss 11.565825448081501 valid acc 2/16
Epoch 13 loss 11.495440687194618 valid acc 2/16
Epoch 13 loss 11.490024860757515 valid acc 2/16
Epoch 13 loss 11.451354951443987 valid acc 2/16
Epoch 13 loss 11.475999046629509 valid acc 2/16
Epoch 13 loss 11.440542337323127 valid acc 2/16
Epoch 13 loss 11.512018495643204 valid acc 2/16
Epoch 13 loss 11.52990324598154 valid acc 2/16
Epoch 13 loss 11.52385166511698 valid acc 2/16
Epoch 13 loss 11.53959571558219 valid acc 2/16
Epoch 13 loss 11.514973007660664 valid acc 2/16
Epoch 13 loss 11.467063111241536 valid acc 2/16
Epoch 13 loss 11.483517833249843 valid acc 2/16
Epoch 13 loss 11.510364339742084 valid acc 2/16
Epoch 13 loss 11.529339743726451 valid acc 2/16
Epoch 13 loss 11.539057563882587 valid acc 2/16
Epoch 13 loss 11.516342655492743 valid acc 2/16
Epoch 13 loss 11.483788772938777 valid acc 2/16
Epoch 13 loss 11.481246695371112 valid acc 2/16
Epoch 13 loss 11.565833745470176 valid acc 2/16
Epoch 13 loss 11.519277577634302 valid acc 2/16
Epoch 13 loss 11.433546812478289 valid acc 2/16
Epoch 13 loss 11.533510702608869 valid acc 2/16
Epoch 13 loss 11.524836336835449 valid acc 2/16
Epoch 13 loss 11.476328933009729 valid acc 2/16
Epoch 13 loss 11.485012363944218 valid acc 2/16
Epoch 13 loss 11.451186118298136 valid acc 2/16
Epoch 13 loss 11.570103891295819 valid acc 2/16
Epoch 13 loss 11.522663766163557 valid acc 2/16
Epoch 13 loss 11.46318463594333 valid acc 2/16
Epoch 13 loss 11.552376860999157 valid acc 2/16
Epoch 13 loss 11.480934706866933 valid acc 2/16
Epoch 13 loss 11.49102597394271 valid acc 2/16
Epoch 13 loss 11.499255016975251 valid acc 2/16
Epoch 13 loss 11.500635945899035 valid acc 2/16
Epoch 13 loss 11.511968116535126 valid acc 2/16
Epoch 13 loss 11.590750035703055 valid acc 2/16
Epoch 13 loss 11.461776776211652 valid acc 2/16
Epoch 13 loss 11.475448057211462 valid acc 2/16
Epoch 13 loss 11.52006732082873 valid acc 2/16
Epoch 13 loss 11.495736932062385 valid acc 2/16
Epoch 13 loss 11.526099386110399 valid acc 2/16
Epoch 13 loss 11.46151622076149 valid acc 2/16
Epoch 13 loss 11.483995891152018 valid acc 2/16
Epoch 13 loss 11.51700244145768 valid acc 2/16
Epoch 13 loss 11.484266632225557 valid acc 2/16
Epoch 13 loss 11.490201847388168 valid acc 2/16
Epoch 13 loss 11.452151022172394 valid acc 2/16
Epoch 13 loss 11.559424873959617 valid acc 2/16
Epoch 13 loss 11.55497582924023 valid acc 2/16
Epoch 13 loss 11.425903818618362 valid acc 2/16
Epoch 13 loss 11.508925532305973 valid acc 2/16
Epoch 13 loss 11.45196667559854 valid acc 2/16
Epoch 14 loss 2.2855129852615432 valid acc 2/16
Epoch 14 loss 11.474020865071306 valid acc 2/16
Epoch 14 loss 11.468779602240197 valid acc 2/16
Epoch 14 loss 11.509655350799008 valid acc 2/16
Epoch 14 loss 11.537985731500193 valid acc 2/16
Epoch 14 loss 11.418688784913623 valid acc 2/16
Epoch 14 loss 11.496448729028979 valid acc 2/16
Epoch 14 loss 11.55043017729365 valid acc 2/16
Epoch 14 loss 11.488483103229443 valid acc 2/16
Epoch 14 loss 11.509890957639442 valid acc 2/16
Epoch 14 loss 11.568857068887722 valid acc 2/16
Epoch 14 loss 11.49428410713156 valid acc 2/16
Epoch 14 loss 11.48864440802869 valid acc 2/16
Epoch 14 loss 11.449341934346052 valid acc 2/16
Epoch 14 loss 11.475486847974397 valid acc 2/16
Epoch 14 loss 11.438820338122651 valid acc 2/16
Epoch 14 loss 11.51358611213906 valid acc 2/16
Epoch 14 loss 11.52931553036543 valid acc 2/16
Epoch 14 loss 11.52432563902868 valid acc 2/16
Epoch 14 loss 11.540772566471588 valid acc 2/16
Epoch 14 loss 11.51507019896617 valid acc 2/16
Epoch 14 loss 11.466011341992221 valid acc 2/16
Epoch 14 loss 11.483526801043219 valid acc 2/16
Epoch 14 loss 11.510710019256276 valid acc 2/16
Epoch 14 loss 11.529693728977913 valid acc 2/16
Epoch 14 loss 11.540303574792123 valid acc 2/16
Epoch 14 loss 11.516532750738744 valid acc 2/16
Epoch 14 loss 11.48302896771084 valid acc 2/16
Epoch 14 loss 11.480369582202728 valid acc 2/16
Epoch 14 loss 11.56816585814614 valid acc 2/16
Epoch 14 loss 11.52024859378136 valid acc 2/16
Epoch 14 loss 11.43180146729256 valid acc 2/16
Epoch 14 loss 11.533310671901859 valid acc 2/16
Epoch 14 loss 11.525270980838943 valid acc 2/16
Epoch 14 loss 11.475759607227996 valid acc 2/16
Epoch 14 loss 11.484464379737965 valid acc 2/16
Epoch 14 loss 11.449642104372757 valid acc 2/16
Epoch 14 loss 11.571611656401448 valid acc 2/16
Epoch 14 loss 11.523466798447542 valid acc 2/16
Epoch 14 loss 11.463331807102021 valid acc 2/16
Epoch 14 loss 11.553735202680347 valid acc 2/16
Epoch 14 loss 11.480077030491762 valid acc 2/16
Epoch 14 loss 11.49102532091549 valid acc 2/16
Epoch 14 loss 11.498623675312412 valid acc 2/16
Epoch 14 loss 11.501016977922223 valid acc 2/16
Epoch 14 loss 11.512423781494055 valid acc 2/16
Epoch 14 loss 11.593024589831035 valid acc 2/16
Epoch 14 loss 11.461316194538071 valid acc 2/16
Epoch 14 loss 11.474070638992995 valid acc 2/16
Epoch 14 loss 11.520776056478041 valid acc 2/16
Epoch 14 loss 11.496152068453995 valid acc 2/16
Epoch 14 loss 11.527252659495112 valid acc 2/16
Epoch 14 loss 11.461050464892718 valid acc 2/16
Epoch 14 loss 11.48247402521606 valid acc 2/16
Epoch 14 loss 11.51641627907572 valid acc 2/16
Epoch 14 loss 11.48348321552061 valid acc 2/16
Epoch 14 loss 11.490174933099283 valid acc 2/16
Epoch 14 loss 11.450257982183928 valid acc 2/16
Epoch 14 loss 11.560765945969163 valid acc 2/16
Epoch 14 loss 11.556790057870899 valid acc 2/16
Epoch 14 loss 11.423929399742969 valid acc 2/16
Epoch 14 loss 11.50880785607752 valid acc 2/16
Epoch 14 loss 11.450889638838525 valid acc 2/16
Epoch 15 loss 2.2850803636095973 valid acc 2/16
Epoch 15 loss 11.473381122739537 valid acc 2/16
Epoch 15 loss 11.468075615630935 valid acc 2/16
Epoch 15 loss 11.509133286299283 valid acc 2/16
Epoch 15 loss 11.538852441486352 valid acc 2/16
Epoch 15 loss 11.417793180052161 valid acc 2/16
Epoch 15 loss 11.496191993833733 valid acc 2/16
Epoch 15 loss 11.551693566030483 valid acc 2/16
Epoch 15 loss 11.487902207828272 valid acc 2/16
Epoch 15 loss 11.510071239443885 valid acc 2/16
Epoch 15 loss 11.571111834418023 valid acc 2/16
Epoch 15 loss 11.493449418810444 valid acc 2/16
Epoch 15 loss 11.487709892434719 valid acc 2/16
Epoch 15 loss 11.447886758586822 valid acc 2/16
Epoch 15 loss 11.475156907653323 valid acc 2/16
Epoch 15 loss 11.437547408236105 valid acc 2/16
Epoch 15 loss 11.514769562348771 valid acc 2/16
Epoch 15 loss 11.528908626562867 valid acc 2/16
Epoch 15 loss 11.52469115425069 valid acc 2/16
Epoch 15 loss 11.541661328552138 valid acc 2/16
Epoch 15 loss 11.515118871338771 valid acc 2/16
Epoch 15 loss 11.465246882597686 valid acc 2/16
Epoch 15 loss 11.483517492776295 valid acc 2/16
Epoch 15 loss 11.510997625099895 valid acc 2/16
Epoch 15 loss 11.529922986647595 valid acc 2/16
Epoch 15 loss 11.541248253381388 valid acc 2/16
Epoch 15 loss 11.516693970971021 valid acc 2/16
Epoch 15 loss 11.482496276020258 valid acc 2/16
Epoch 15 loss 11.479731006698405 valid acc 2/16
Epoch 15 loss 11.56989488566875 valid acc 2/16
Epoch 15 loss 11.520998453850376 valid acc 2/16
Epoch 15 loss 11.43053216004009 valid acc 2/16
Epoch 15 loss 11.533162429008422 valid acc 2/16
Epoch 15 loss 11.525604200641133 valid acc 2/16
Epoch 15 loss 11.475367408557318 valid acc 2/16
Epoch 15 loss 11.4840623566273 valid acc 2/16
Epoch 15 loss 11.44851173752615 valid acc 2/16
Epoch 15 loss 11.572742781601082 valid acc 2/16
Epoch 15 loss 11.524065173282342 valid acc 2/16
Epoch 15 loss 11.46344483224932 valid acc 2/16
Epoch 15 loss 11.554752238171563 valid acc 2/16
Epoch 15 loss 11.47948301624077 valid acc 2/16
Epoch 15 loss 11.49100155807344 valid acc 2/16
Epoch 15 loss 11.498174449058514 valid acc 2/16
Epoch 15 loss 11.501324409169818 valid acc 2/16
Epoch 15 loss 11.512758582670838 valid acc 2/16
Epoch 15 loss 11.594705463448854 valid acc 2/16
Epoch 15 loss 11.460976166424214 valid acc 2/16
Epoch 15 loss 11.47306789152583 valid acc 2/16
Epoch 15 loss 11.521325171932425 valid acc 2/16
Epoch 15 loss 11.496470779494 valid acc 2/16
Epoch 15 loss 11.528107175697123 valid acc 2/16
Epoch 15 loss 11.46073094022259 valid acc 2/16
Epoch 15 loss 11.481350246579115 valid acc 2/16
Epoch 15 loss 11.516021922770845 valid acc 2/16
Epoch 15 loss 11.482916746304848 valid acc 2/16
Epoch 15 loss 11.490175061746966 valid acc 2/16
Epoch 15 loss 11.448877571938688 valid acc 2/16
Epoch 15 loss 11.561756394808596 valid acc 2/16
Epoch 15 loss 11.558129981573266 valid acc 2/16
Epoch 15 loss 11.422495416458425 valid acc 2/16
Epoch 15 loss 11.5087215290408 valid acc 2/16
Epoch 15 loss 11.450126980021489 valid acc 2/16
Epoch 16 loss 2.2847670876919324 valid acc 2/16
Epoch 16 loss 11.472898820140886 valid acc 2/16
Epoch 16 loss 11.467569089973843 valid acc 2/16
Epoch 16 loss 11.508750624082383 valid acc 2/16
Epoch 16 loss 11.53949961822914 valid acc 2/16
Epoch 16 loss 11.417155252579725 valid acc 2/16
Epoch 16 loss 11.496006489554457 valid acc 2/16
Epoch 16 loss 11.552664098143055 valid acc 2/16
Epoch 16 loss 11.487505352384987 valid acc 2/16
Epoch 16 loss 11.510196871767283 valid acc 2/16
Epoch 16 loss 11.57278596097579 valid acc 2/16
Epoch 16 loss 11.492843376033719 valid acc 2/16
Epoch 16 loss 11.487079183774798 valid acc 2/16
Epoch 16 loss 11.446830778531771 valid acc 2/16
Epoch 16 loss 11.474945114919581 valid acc 2/16
Epoch 16 loss 11.436601973708921 valid acc 2/16
Epoch 16 loss 11.51565970816756 valid acc 2/16
Epoch 16 loss 11.528624610807881 valid acc 2/16
Epoch 16 loss 11.524969172203996 valid acc 2/16
Epoch 16 loss 11.542329055470493 valid acc 2/16
Epoch 16 loss 11.515133936792882 valid acc 2/16
Epoch 16 loss 11.464687553706156 valid acc 2/16
Epoch 16 loss 11.483494739046364 valid acc 2/16
Epoch 16 loss 11.51123091074659 valid acc 2/16
Epoch 16 loss 11.53006478688928 valid acc 2/16
Epoch 16 loss 11.541961174923609 valid acc 2/16
Epoch 16 loss 11.516825402395018 valid acc 2/16
Epoch 16 loss 11.48212045313879 valid acc 2/16
Epoch 16 loss 11.479262912875342 valid acc 2/16
Epoch 16 loss 11.571174260529787 valid acc 2/16
Epoch 16 loss 11.521573548531983 valid acc 2/16
Epoch 16 loss 11.429605672123694 valid acc 2/16
Epoch 16 loss 11.533049403289228 valid acc 2/16
Epoch 16 loss 11.525856175761714 valid acc 2/16
Epoch 16 loss 11.475095388751722 valid acc 2/16
Epoch 16 loss 11.483764079397135 valid acc 2/16
Epoch 16 loss 11.447680482756459 valid acc 2/16
Epoch 16 loss 11.573588845518579 valid acc 2/16
Epoch 16 loss 11.524508229243414 valid acc 2/16
Epoch 16 loss 11.463528399445451 valid acc 2/16
Epoch 16 loss 11.555511270887678 valid acc 2/16
Epoch 16 loss 11.479070655475919 valid acc 2/16
Epoch 16 loss 11.490963630387117 valid acc 2/16
Epoch 16 loss 11.497852295370617 valid acc 2/16
Epoch 16 loss 11.501568151750366 valid acc 2/16
Epoch 16 loss 11.513001578453057 valid acc 2/16
Epoch 16 loss 11.595945878018341 valid acc 2/16
Epoch 16 loss 11.460722061685058 valid acc 2/16
Epoch 16 loss 11.472334879863855 valid acc 2/16
Epoch 16 loss 11.521747229449613 valid acc 2/16
Epoch 16 loss 11.496712469116456 valid acc 2/16
Epoch 16 loss 11.528738000549536 valid acc 2/16
Epoch 16 loss 11.460510307180776 valid acc 2/16
Epoch 16 loss 11.48051747339592 valid acc 2/16
Epoch 16 loss 11.51575674633213 valid acc 2/16
Epoch 16 loss 11.482504668411337 valid acc 2/16
Epoch 16 loss 11.490187507976056 valid acc 2/16
Epoch 16 loss 11.447868018605222 valid acc 2/16
Epoch 16 loss 11.56248595365987 valid acc 2/16
Epoch 16 loss 11.55911758208625 valid acc 2/16
Epoch 16 loss 11.421451381204479 valid acc 2/16
Epoch 16 loss 11.50865593494944 valid acc 2/16
Epoch 16 loss 11.44958516780483 valid acc 2/16
Epoch 17 loss 2.2845397507862955 valid acc 2/16
Epoch 17 loss 11.47253260990213 valid acc 2/16
Epoch 17 loss 11.467202338303748 valid acc 2/16
Epoch 17 loss 11.508467585801323 valid acc 2/16
Epoch 17 loss 11.53998105446567 valid acc 2/16
Epoch 17 loss 11.416698882801853 valid acc 2/16
Epoch 17 loss 11.495869986672673 valid acc 2/16
Epoch 17 loss 11.553406964705339 valid acc 2/16
Epoch 17 loss 11.487233855090476 valid acc 2/16
Epoch 17 loss 11.510281912494733 valid acc 2/16
Epoch 17 loss 11.574027632423487 valid acc 2/16
Epoch 17 loss 11.492401324083055 valid acc 2/16
Epoch 17 loss 11.486655365310533 valid acc 2/16
Epoch 17 loss 11.446062124511803 valid acc 2/16
Epoch 17 loss 11.47481012566548 valid acc 2/16
Epoch 17 loss 11.435897372425986 valid acc 2/16
Epoch 17 loss 11.516327625531613 valid acc 2/16
Epoch 17 loss 11.528425070995008 valid acc 2/16
Epoch 17 loss 11.525178702170024 valid acc 2/16
Epoch 17 loss 11.542829053783422 valid acc 2/16
Epoch 17 loss 11.51512772779796 valid acc 2/16
Epoch 17 loss 11.464276287323214 valid acc 2/16
Epoch 17 loss 11.483463997137022 valid acc 2/16
Epoch 17 loss 11.511417078696 valid acc 2/16
Epoch 17 loss 11.530147286211935 valid acc 2/16
Epoch 17 loss 11.542497512635459 valid acc 2/16
Epoch 17 loss 11.516930021111069 valid acc 2/16
Epoch 17 loss 11.48185398817411 valid acc 2/16
Epoch 17 loss 11.478918141946895 valid acc 2/16
Epoch 17 loss 11.572119861613881 valid acc 2/16
Epoch 17 loss 11.52201246067988 valid acc 2/16
Epoch 17 loss 11.428927512168782 valid acc 2/16
Epoch 17 loss 11.532961709213268 valid acc 2/16
Epoch 17 loss 11.526044953499119 valid acc 2/16
Epoch 17 loss 11.474905681567327 valid acc 2/16
Epoch 17 loss 11.483540998576201 valid acc 2/16
Epoch 17 loss 11.447067125668777 valid acc 2/16
Epoch 17 loss 11.574220523816834 valid acc 2/16
Epoch 17 loss 11.524834801075414 valid acc 2/16
Epoch 17 loss 11.463588546310405 valid acc 2/16
Epoch 17 loss 11.55607650630188 valid acc 2/16
Epoch 17 loss 11.478783985748649 valid acc 2/16
Epoch 17 loss 11.49091890520868 valid acc 2/16
Epoch 17 loss 11.497619914105647 valid acc 2/16
Epoch 17 loss 11.501759088724956 valid acc 2/16
Epoch 17 loss 11.51317635256278 valid acc 2/16
Epoch 17 loss 11.596860504621732 valid acc 2/16
Epoch 17 loss 11.46053050895572 valid acc 2/16
Epoch 17 loss 11.471797257335352 valid acc 2/16
Epoch 17 loss 11.522069786918934 valid acc 2/16
Epoch 17 loss 11.496894250589316 valid acc 2/16
Epoch 17 loss 11.529202484063392 valid acc 2/16
Epoch 17 loss 11.460357265075647 valid acc 2/16
Epoch 17 loss 11.479898743951438 valid acc 2/16
Epoch 17 loss 11.51557878606088 valid acc 2/16
Epoch 17 loss 11.482203463051743 valid acc 2/16
Epoch 17 loss 11.490204550752381 valid acc 2/16
Epoch 17 loss 11.447127979193077 valid acc 2/16
Epoch 17 loss 11.563022432306443 valid acc 2/16
Epoch 17 loss 11.55984449426876 valid acc 2/16
Epoch 17 loss 11.420689817191967 valid acc 2/16
Epoch 17 loss 11.508604921606333 valid acc 2/16
Epoch 17 loss 11.449199291809588 valid acc 2/16
Epoch 18 loss 2.2843745257624954 valid acc 2/16
Epoch 18 loss 11.472253249365146 valid acc 2/16
Epoch 18 loss 11.466935491440315 valid acc 2/16
Epoch 18 loss 11.508256833025548 valid acc 2/16
Epoch 18 loss 11.540338373447725 valid acc 2/16
Epoch 18 loss 11.416371252426615 valid acc 2/16
Epoch 18 loss 11.49576815323956 valid acc 2/16
Epoch 18 loss 11.553974070212513 valid acc 2/16
Epoch 18 loss 11.487048083636012 valid acc 2/16
Epoch 18 loss 11.51033785919903 valid acc 2/16
Epoch 18 loss 11.574947968860108 valid acc 2/16
Epoch 18 loss 11.492077770885789 valid acc 2/16
Epoch 18 loss 11.486372218719673 valid acc 2/16
Epoch 18 loss 11.445501224438033 valid acc 2/16
Epoch 18 loss 11.474725069714848 valid acc 2/16
Epoch 18 loss 11.435370967653737 valid acc 2/16
Epoch 18 loss 11.516828016669358 valid acc 2/16
Epoch 18 loss 11.528284147394187 valid acc 2/16
Epoch 18 loss 11.52533563575107 valid acc 2/16
Epoch 18 loss 11.543202685356581 valid acc 2/16
Epoch 18 loss 11.515109401041638 valid acc 2/16
Epoch 18 loss 11.463972785235844 valid acc 2/16
Epoch 18 loss 11.483429805306978 valid acc 2/16
Epoch 18 loss 11.511564032794027 valid acc 2/16
Epoch 18 loss 11.530190850561873 valid acc 2/16
Epoch 18 loss 11.542900149978674 valid acc 2/16
Epoch 18 loss 11.517012039319257 valid acc 2/16
Epoch 18 loss 11.481664324470072 valid acc 2/16
Epoch 18 loss 11.478663375351793 valid acc 2/16
Epoch 18 loss 11.57281840525977 valid acc 2/16
Epoch 18 loss 11.52234625705566 valid acc 2/16
Epoch 18 loss 11.428430040131492 valid acc 2/16
Epoch 18 loss 11.53289303087434 valid acc 2/16
Epoch 18 loss 11.526185484543833 valid acc 2/16
Epoch 18 loss 11.474772786901626 valid acc 2/16
Epoch 18 loss 11.48337323469348 valid acc 2/16
Epoch 18 loss 11.446613397130427 valid acc 2/16
Epoch 18 loss 11.574691634716153 valid acc 2/16
Epoch 18 loss 11.525074725060561 valid acc 2/16
Epoch 18 loss 11.463630941603332 valid acc 2/16
Epoch 18 loss 11.55649680019091 valid acc 2/16
Epoch 18 loss 11.47858455365175 valid acc 2/16
Epoch 18 loss 11.490872663420365 valid acc 2/16
Epoch 18 loss 11.497451566985065 valid acc 2/16
Epoch 18 loss 11.501907378545294 valid acc 2/16
Epoch 18 loss 11.513301190078185 valid acc 2/16
Epoch 18 loss 11.59753463482594 valid acc 2/16
Epoch 18 loss 11.460385235618102 valid acc 2/16
Epoch 18 loss 11.471401866952787 valid acc 2/16
Epoch 18 loss 11.522315280704847 valid acc 2/16
Epoch 18 loss 11.497030216870577 valid acc 2/16
Epoch 18 loss 11.529543863345593 valid acc 2/16
Epoch 18 loss 11.460250798397354 valid acc 2/16
Epoch 18 loss 11.479438170909262 valid acc 2/16
Epoch 18 loss 11.515459765458797 valid acc 2/16
Epoch 18 loss 11.481982453707994 valid acc 2/16
Epoch 18 loss 11.490222272113167 valid acc 2/16
Epoch 18 loss 11.4465844992345 valid acc 2/16
Epoch 18 loss 11.563416528040909 valid acc 2/16
Epoch 18 loss 11.560379040217647 valid acc 2/16
Epoch 18 loss 11.420133483306703 valid acc 2/16
Epoch 18 loss 11.508564698094128 valid acc 2/16
Epoch 18 loss 11.448923955104258 valid acc 2/16
Epoch 19 loss 2.284254313415164 valid acc 2/16
Epoch 19 loss 11.472039523565936 valid acc 2/16
Epoch 19 loss 11.466740598218509 valid acc 2/16
Epoch 19 loss 11.508099143446554 valid acc 2/16
Epoch 19 loss 11.540603233318024 valid acc 2/16
Epoch 19 loss 11.41613537167142 valid acc 2/16
Epoch 19 loss 11.495691421571088 valid acc 2/16
Epoch 19 loss 11.554406147494525 valid acc 2/16
Epoch 19 loss 11.486921083630897 valid acc 2/16
Epoch 19 loss 11.510373522713524 valid acc 2/16
Epoch 19 loss 11.575629901541635 valid acc 2/16
Epoch 19 loss 11.491840331320205 valid acc 2/16
Epoch 19 loss 11.486184465994961 valid acc 2/16
Epoch 19 loss 11.445091088488015 valid acc 2/16
Epoch 19 loss 11.474672390762517 valid acc 2/16
Epoch 19 loss 11.434977005177169 valid acc 2/16
Epoch 19 loss 11.51720254011151 valid acc 2/16
Epoch 19 loss 11.528184205977205 valid acc 2/16
Epoch 19 loss 11.525452678919939 valid acc 2/16
Epoch 19 loss 11.543481555199326 valid acc 2/16
Epoch 19 loss 11.515085273708491 valid acc 2/16
Epoch 19 loss 11.46374820924526 valid acc 2/16
Epoch 19 loss 11.483395417813751 valid acc 2/16
Epoch 19 loss 11.511679157157602 valid acc 2/16
Epoch 19 loss 11.530209737604867 valid acc 2/16
Epoch 19 loss 11.543201987623961 valid acc 2/16
Epoch 19 loss 11.517075695870409 valid acc 2/16
Epoch 19 loss 11.481528912590461 valid acc 2/16
Epoch 19 loss 11.47847471350272 valid acc 2/16
Epoch 19 loss 11.573334393016125 valid acc 2/16
Epoch 19 loss 11.522599450505645 valid acc 2/16
Epoch 19 loss 11.428064491040328 valid acc 2/16
Epoch 19 loss 11.532839037282102 valid acc 2/16
Epoch 19 loss 11.526289637444474 valid acc 2/16
Epoch 19 loss 11.47467934893194 valid acc 2/16
Epoch 19 loss 11.48324660565455 valid acc 2/16
Epoch 19 loss 11.446277106449575 valid acc 2/16
Epoch 19 loss 11.575042803708602 valid acc 2/16
Epoch 19 loss 11.525250564640057 valid acc 2/16
Epoch 19 loss 11.46366029934492 valid acc 2/16
Epoch 19 loss 11.556809011503468 valid acc 2/16
Epoch 19 loss 11.478445796684976 valid acc 2/16
Epoch 19 loss 11.490828306312556 valid acc 2/16
Epoch 19 loss 11.497329229505176 valid acc 2/16
Epoch 19 loss 11.502021814775695 valid acc 2/16
Epoch 19 loss 11.513389869200857 valid acc 2/16
Epoch 19 loss 11.598031447191275 valid acc 2/16
Epoch 19 loss 11.460274610415441 valid acc 2/16
Epoch 19 loss 11.471110423206632 valid acc 2/16
Epoch 19 loss 11.522501545933912 valid acc 2/16
Epoch 19 loss 11.497131533508119 valid acc 2/16
Epoch 19 loss 11.529794442736392 valid acc 2/16
Epoch 19 loss 11.460176618760178 valid acc 2/16
Epoch 19 loss 11.479094851933922 valid acc 2/16
Epoch 19 loss 11.515380557869602 valid acc 2/16
Epoch 19 loss 11.48181978405989 valid acc 2/16
Epoch 19 loss 11.490238808169956 valid acc 2/16
Epoch 19 loss 11.446184770995437 valid acc 2/16
Epoch 19 loss 11.563705873654872 valid acc 2/16
Epoch 19 loss 11.560771898111648 valid acc 2/16
Epoch 19 loss 11.419726601375412 valid acc 2/16
Epoch 19 loss 11.508532758246918 valid acc 2/16
Epoch 19 loss 11.448727214355383 valid acc 2/16
Epoch 20 loss 2.284166786095832 valid acc 2/16
Epoch 20 loss 11.47187573913452 valid acc 2/16
Epoch 20 loss 11.466597835504427 valid acc 2/16
Epoch 20 loss 11.507980747195258 valid acc 2/16
Epoch 20 loss 11.540799446775077 valid acc 2/16
Epoch 20 loss 11.415965143326051 valid acc 2/16
Epoch 20 loss 11.49563319513277 valid acc 2/16
Epoch 20 loss 11.554734853724577 valid acc 2/16
Epoch 20 loss 11.486834427527198 valid acc 2/16
Epoch 20 loss 11.510395381714485 valid acc 2/16
Epoch 20 loss 11.576135126961349 valid acc 2/16
Epoch 20 loss 11.491665739995529 valid acc 2/16
Epoch 20 loss 11.486061158226965 valid acc 2/16
Epoch 20 loss 11.444790678010651 valid acc 2/16
Epoch 20 loss 11.474640585313777 valid acc 2/16
Epoch 20 loss 11.434681796910331 valid acc 2/16
Epoch 20 loss 11.517482701217585 valid acc 2/16
Epoch 20 loss 11.528113092822913 valid acc 2/16
Epoch 20 loss 11.525539720860326 valid acc 2/16
Epoch 20 loss 11.543689574533715 valid acc 2/16
Epoch 20 loss 11.515059434030016 valid acc 2/16
Epoch 20 loss 11.463581707623621 valid acc 2/16
Epoch 20 loss 11.48336291459612 valid acc 2/16
Epoch 20 loss 11.511768857768498 valid acc 2/16
Epoch 20 loss 11.530213656756217 valid acc 2/16
Epoch 20 loss 11.543428050299802 valid acc 2/16
Epoch 20 loss 11.517124765238087 valid acc 2/16
Epoch 20 loss 11.481431999257786 valid acc 2/16
Epoch 20 loss 11.478334816786772 valid acc 2/16
Epoch 20 loss 11.573715613798257 valid acc 2/16
Epoch 20 loss 11.52279112688301 valid acc 2/16
Epoch 20 loss 11.42779551495875 valid acc 2/16
Epoch 20 loss 11.532796571215144 valid acc 2/16
Epoch 20 loss 11.52636659177896 valid acc 2/16
Epoch 20 loss 11.474613451607205 valid acc 2/16
Epoch 20 loss 11.483150799163173 valid acc 2/16
Epoch 20 loss 11.44602749019261 valid acc 2/16
Epoch 20 loss 11.575304522868215 valid acc 2/16
Epoch 20 loss 11.525379202035964 valid acc 2/16
Epoch 20 loss 11.463680297323592 valid acc 2/16
Epoch 20 loss 11.557040787585107 valid acc 2/16
Epoch 20 loss 11.478349294159866 valid acc 2/16
Epoch 20 loss 11.490787777567192 valid acc 2/16
Epoch 20 loss 11.497240130854781 valid acc 2/16
Epoch 20 loss 11.502109696151864 valid acc 2/16
Epoch 20 loss 11.513452576807834 valid acc 2/16
Epoch 20 loss 11.5983976136392 valid acc 2/16
Epoch 20 loss 11.460190144436627 valid acc 2/16
Epoch 20 loss 11.470895189861272 valid acc 2/16
Epoch 20 loss 11.52264254050368 valid acc 2/16
Epoch 20 loss 11.497206839701924 valid acc 2/16
Epoch 20 loss 11.529978208137013 valid acc 2/16
Epoch 20 loss 11.460124914656104 valid acc 2/16
Epoch 20 loss 11.478838676455089 valid acc 2/16
Epoch 20 loss 11.51532819661448 valid acc 2/16
Epoch 20 loss 11.481699748156748 valid acc 2/16
Epoch 20 loss 11.4902533908168 valid acc 2/16
Epoch 20 loss 11.44589040577667 valid acc 2/16
Epoch 20 loss 11.563918270121128 valid acc 2/16
Epoch 20 loss 11.56106052526428 valid acc 2/16
Epoch 20 loss 11.419428744872501 valid acc 2/16
Epoch 20 loss 11.508507328667047 valid acc 2/16
Epoch 20 loss 11.44858648616402 valid acc 2/16
Epoch 21 loss 2.2841030258230406 valid acc 2/16
Epoch 21 loss 11.471750122142534 valid acc 2/16
Epoch 21 loss 11.466493015467105 valid acc 2/16
Epoch 21 loss 11.507891634550525 valid acc 2/16
Epoch 21 loss 11.540944795967967 valid acc 2/16
Epoch 21 loss 11.415842046358378 valid acc 2/16
Epoch 21 loss 11.495588796848814 valid acc 2/16
Epoch 21 loss 11.554984628386217 valid acc 2/16
Epoch 21 loss 11.486775471176166 valid acc 2/16
Epoch 21 loss 11.51040806177741 valid acc 2/16
Epoch 21 loss 11.576509447492494 valid acc 2/16
Epoch 21 loss 11.491537165911183 valid acc 2/16
Epoch 21 loss 11.485981169755084 valid acc 2/16
Epoch 21 loss 11.444570313764316 valid acc 2/16
Epoch 21 loss 11.474622112929193 valid acc 2/16
Epoch 21 loss 11.43446039382304 valid acc 2/16
Epoch 21 loss 11.517692217482516 valid acc 2/16
Epoch 21 loss 11.528062357539667 valid acc 2/16
Epoch 21 loss 11.52560432729546 valid acc 2/16
Epoch 21 loss 11.543844717149994 valid acc 2/16
Epoch 21 loss 11.515034348173169 valid acc 2/16
Epoch 21 loss 11.463458084291053 valid acc 2/16
Epoch 21 loss 11.483333458821017 valid acc 2/16
Epoch 21 loss 11.511838469973144 valid acc 2/16
Epoch 21 loss 11.530209048566844 valid acc 2/16
Epoch 21 loss 11.543597261724056 valid acc 2/16
Epoch 21 loss 11.517162411296914 valid acc 2/16
Epoch 21 loss 11.481362503631331 valid acc 2/16
Epoch 21 loss 11.478231000338354 valid acc 2/16
Epoch 21 loss 11.573997385872588 valid acc 2/16
Epoch 21 loss 11.522936012751313 valid acc 2/16
Epoch 21 loss 11.427597379015877 valid acc 2/16
Epoch 21 loss 11.53276322565607 valid acc 2/16
Epoch 21 loss 11.526423328974499 valid acc 2/16
Epoch 21 loss 11.474566855762097 valid acc 2/16
Epoch 21 loss 11.483078208606383 valid acc 2/16
Epoch 21 loss 11.445841997813478 valid acc 2/16
Epoch 21 loss 11.575499593739702 valid acc 2/16
Epoch 21 loss 11.525473175736895 valid acc 2/16
Epoch 21 loss 11.463693694823082 valid acc 2/16
Epoch 21 loss 11.557212784529042 valid acc 2/16
Epoch 21 loss 11.478282235851525 valid acc 2/16
Epoch 21 loss 11.490751988292535 valid acc 2/16
Epoch 21 loss 11.497175140796788 valid acc 2/16
Epoch 21 loss 11.502176925564445 valid acc 2/16
Epoch 21 loss 11.513496745422763 valid acc 2/16
Epoch 21 loss 11.598667552278119 valid acc 2/16
Epoch 21 loss 11.460125543463365 valid acc 2/16
Epoch 21 loss 11.470735978832575 valid acc 2/16
Epoch 21 loss 11.522749072647114 valid acc 2/16
Epoch 21 loss 11.497262718949695 valid acc 2/16
Epoch 21 loss 11.530112891511774 valid acc 2/16
Epoch 21 loss 11.46008889767973 valid acc 2/16
Epoch 21 loss 11.478647382449164 valid acc 2/16
Epoch 21 loss 11.515293883599993 valid acc 2/16
Epoch 21 loss 11.481610982601598 valid acc 2/16
Epoch 21 loss 11.490265822058799 valid acc 2/16
Epoch 21 loss 11.44567340326827 valid acc 2/16
Epoch 21 loss 11.564074189553736 valid acc 2/16
Epoch 21 loss 11.561272539491927 valid acc 2/16
Epoch 21 loss 11.41921053205224 valid acc 2/16
Epoch 21 loss 11.50848708280204 valid acc 2/16
Epoch 21 loss 11.448485746029174 valid acc 2/16
Epoch 22 loss 2.2840565650498523 valid acc 2/16
Epoch 22 loss 11.471653752276078 valid acc 2/16
Epoch 22 loss 11.466415910594343 valid acc 2/16
Epoch 22 loss 11.507824447169122 valid acc 2/16
Epoch 22 loss 11.541052497592412 valid acc 2/16
Epoch 22 loss 11.415752874785529 valid acc 2/16
Epoch 22 loss 11.495554833736975 valid acc 2/16
Epoch 22 loss 11.555174249716421 valid acc 2/16
Epoch 22 loss 11.486735518588656 valid acc 2/16
Epoch 22 loss 11.510414789579013 valid acc 2/16
Epoch 22 loss 11.57678681984055 valid acc 2/16
Epoch 22 loss 11.491442368115504 valid acc 2/16
Epoch 22 loss 11.485930113952204 valid acc 2/16
Epoch 22 loss 11.444408458299574 valid acc 2/16
Epoch 22 loss 11.47461204120118 valid acc 2/16
Epoch 22 loss 11.434294240557364 valid acc 2/16
Epoch 22 loss 11.517848889706414 valid acc 2/16
Epoch 22 loss 11.528026083035403 valid acc 2/16
Epoch 22 loss 11.525652221048336 valid acc 2/16
Epoch 22 loss 11.543960437326874 valid acc 2/16
Epoch 22 loss 11.515011365097582 valid acc 2/16
Epoch 22 loss 11.463366199346723 valid acc 2/16
Epoch 22 loss 11.483307562445573 valid acc 2/16
Epoch 22 loss 11.511892328861329 valid acc 2/16
Epoch 22 loss 11.53020006714307 valid acc 2/16
Epoch 22 loss 11.543723875588421 valid acc 2/16
Epoch 22 loss 11.517191196586273 valid acc 2/16
Epoch 22 loss 11.48131258945597 valid acc 2/16
Epoch 22 loss 11.47815393000274 valid acc 2/16
Epoch 22 loss 11.574205772162758 valid acc 2/16
Epoch 22 loss 11.523045399488641 valid acc 2/16
Epoch 22 loss 11.427451290848504 valid acc 2/16
Epoch 22 loss 11.532737113244334 valid acc 2/16
Epoch 22 loss 11.526465099308412 valid acc 2/16
Epoch 22 loss 11.474533832144287 valid acc 2/16
Epoch 22 loss 11.483023164727115 valid acc 2/16
Epoch 22 loss 11.445704032772486 valid acc 2/16
Epoch 22 loss 11.575645026927862 valid acc 2/16
Epoch 22 loss 11.525541749725907 valid acc 2/16
Epoch 22 loss 11.46370250783819 valid acc 2/16
Epoch 22 loss 11.557340393507712 valid acc 2/16
Epoch 22 loss 11.47823569644809 valid acc 2/16
Epoch 22 loss 11.490721169510369 valid acc 2/16
Epoch 22 loss 11.497127687341756 valid acc 2/16
Epoch 22 loss 11.502228195976134 valid acc 2/16
Epoch 22 loss 11.51352774613501 valid acc 2/16
Epoch 22 loss 11.598866617859493 valid acc 2/16
Epoch 22 loss 11.460076086466106 valid acc 2/16
Epoch 22 loss 11.470618040392097 valid acc 2/16
Epoch 22 loss 11.522829449587938 valid acc 2/16
Epoch 22 loss 11.497304137927848 valid acc 2/16
Epoch 22 loss 11.530211561297296 valid acc 2/16
Epoch 22 loss 11.460063845085426 valid acc 2/16
Epoch 22 loss 11.478504458390784 valid acc 2/16
Epoch 22 loss 11.515271651570945 valid acc 2/16
Epoch 22 loss 11.481545221224096 valid acc 2/16
Epoch 22 loss 11.490276186288545 valid acc 2/16
Epoch 22 loss 11.445513286462926 valid acc 2/16
Epoch 22 loss 11.564188675777963 valid acc 2/16
Epoch 22 loss 11.56142826978692 valid acc 2/16
Epoch 22 loss 11.419050562876754 valid acc 2/16
Epoch 22 loss 11.508470989051911 valid acc 2/16
Epoch 22 loss 11.44841359154291 valid acc 2/16
Epoch 23 loss 2.284022704819737 valid acc 2/16
Epoch 23 loss 11.471579827098726 valid acc 2/16
Epoch 23 loss 11.466359107017976 valid acc 2/16
Epoch 23 loss 11.507773730320078 valid acc 2/16
Epoch 23 loss 11.541132346080662 valid acc 2/16
Epoch 23 loss 11.415688177090454 valid acc 2/16
Epoch 23 loss 11.495528799989252 valid acc 2/16
Epoch 23 loss 11.555318096713659 valid acc 2/16
Epoch 23 loss 11.486708582028147 valid acc 2/16
Epoch 23 loss 11.510417770915254 valid acc 2/16
Epoch 23 loss 11.576992397586753 valid acc 2/16
Epoch 23 loss 11.491372408133818 valid acc 2/16
Epoch 23 loss 11.48589822597525 valid acc 2/16
Epoch 23 loss 11.44428943953206 valid acc 2/16
Epoch 23 loss 11.474607158801856 valid acc 2/16
Epoch 23 loss 11.434169494858397 valid acc 2/16
Epoch 23 loss 11.517966051837234 valid acc 2/16
Epoch 23 loss 11.528000102460679 valid acc 2/16
Epoch 23 loss 11.525687697936238 valid acc 2/16
Epoch 23 loss 11.544046779661677 valid acc 2/16
Epoch 23 loss 11.514991100655232 valid acc 2/16
Epoch 23 loss 11.463297850572301 valid acc 2/16
Epoch 23 loss 11.483285308781078 valid acc 2/16
Epoch 23 loss 11.511933900999393 valid acc 2/16
Epoch 23 loss 11.530189303876805 valid acc 2/16
Epoch 23 loss 11.543818599230562 valid acc 2/16
Epoch 23 loss 11.51721315330892 valid acc 2/16
Epoch 23 loss 11.481276691610711 valid acc 2/16
Epoch 23 loss 11.478096709573926 valid acc 2/16
Epoch 23 loss 11.574359990536252 valid acc 2/16
Epoch 23 loss 11.523127905528884 valid acc 2/16
Epoch 23 loss 11.427343492675442 valid acc 2/16
Epoch 23 loss 11.53271673189188 valid acc 2/16
Epoch 23 loss 11.526495820925792 valid acc 2/16
Epoch 23 loss 11.474510378610278 valid acc 2/16
Epoch 23 loss 11.482981411477216 valid acc 2/16
Epoch 23 loss 11.445601343576396 valid acc 2/16
Epoch 23 loss 11.575753494797542 valid acc 2/16
Epoch 23 loss 11.525591743119133 valid acc 2/16
Epoch 23 loss 11.463708180878895 valid acc 2/16
Epoch 23 loss 11.55743506172545 valid acc 2/16
Epoch 23 loss 11.478203450335034 valid acc 2/16
Epoch 23 loss 11.490695137161776 valid acc 2/16
Epoch 23 loss 11.497093016012942 valid acc 2/16
Epoch 23 loss 11.502267194366212 valid acc 2/16
Epoch 23 loss 11.513549432496006 valid acc 2/16
Epoch 23 loss 11.59901347765317 valid acc 2/16
Epoch 23 loss 11.460038203484897 valid acc 2/16
Epoch 23 loss 11.470530564897421 valid acc 2/16
Epoch 23 loss 11.522890022114257 valid acc 2/16
Epoch 23 loss 11.49733481825788 valid acc 2/16
Epoch 23 loss 11.530283828035596 valid acc 2/16
Epoch 23 loss 11.46004645867065 valid acc 2/16
Epoch 23 loss 11.478397629395152 valid acc 2/16
Epoch 23 loss 11.51525745957341 valid acc 2/16
Epoch 23 loss 11.481496424792127 valid acc 2/16
Epoch 23 loss 11.490284694074866 valid acc 2/16
Epoch 23 loss 11.445395048803054 valid acc 2/16
Epoch 23 loss 11.564272769243694 valid acc 2/16
Epoch 23 loss 11.561542663148934 valid acc 2/16
Epoch 23 loss 11.418933225052513 valid acc 2/16
Epoch 23 loss 11.508458226131788 valid acc 2/16
Epoch 23 loss 11.448361891815697 valid acc 2/16
Epoch 24 loss 2.283998026682622 valid acc 2/16
Epoch 24 loss 11.471523137830724 valid acc 2/16
Epoch 24 loss 11.46631720712239 valid acc 2/16
Epoch 24 loss 11.507735415488606 valid acc 2/16
Epoch 24 loss 11.541191587062512 valid acc 2/16
Epoch 24 loss 11.415641168362528 valid acc 2/16
Epoch 24 loss 11.495508820069762 valid acc 2/16
Epoch 24 loss 11.555427151859359 valid acc 2/16
Epoch 24 loss 11.486690537727869 valid acc 2/16
Epoch 24 loss 11.510418484745376 valid acc 2/16
Epoch 24 loss 11.577144804592582 valid acc 2/16
Epoch 24 loss 11.49132073917671 valid acc 2/16
Epoch 24 loss 11.48587890686347 valid acc 2/16
Epoch 24 loss 11.444201827853838 valid acc 2/16
Epoch 24 loss 11.47460539078864 valid acc 2/16
Epoch 24 loss 11.434075807978855 valid acc 2/16
Epoch 24 loss 11.518053679553907 valid acc 2/16
Epoch 24 loss 11.52798146810196 valid acc 2/16
Epoch 24 loss 11.525713965821197 valid acc 2/16
Epoch 24 loss 11.544111231260123 valid acc 2/16
Epoch 24 loss 11.51497371366117 valid acc 2/16
Epoch 24 loss 11.463246979218885 valid acc 2/16
Epoch 24 loss 11.483266520880594 valid acc 2/16
Epoch 24 loss 11.51196592869583 valid acc 2/16
Epoch 24 loss 11.530178305191058 valid acc 2/16
Epoch 24 loss 11.543889461374732 valid acc 2/16
Epoch 24 loss 11.517229871340454 valid acc 2/16
Epoch 24 loss 11.481250844783588 valid acc 2/16
Epoch 24 loss 11.47805423041024 valid acc 2/16
Epoch 24 loss 11.574474207502298 valid acc 2/16
Epoch 24 loss 11.52319008737074 valid acc 2/16
Epoch 24 loss 11.427263893095208 valid acc 2/16
Epoch 24 loss 11.532700879006294 valid acc 2/16
Epoch 24 loss 11.52651840228123 valid acc 2/16
Epoch 24 loss 11.47449368916341 valid acc 2/16
Epoch 24 loss 11.482949738045413 valid acc 2/16
Epoch 24 loss 11.4455248655366 valid acc 2/16
Epoch 24 loss 11.575834430519196 valid acc 2/16
Epoch 24 loss 11.525628161759386 valid acc 2/16
Epoch 24 loss 11.463711733139696 valid acc 2/16
Epoch 24 loss 11.557505292587596 valid acc 2/16
Epoch 24 loss 11.478181152720708 valid acc 2/16
Epoch 24 loss 11.490673478747253 valid acc 2/16
Epoch 24 loss 11.497067674491324 valid acc 2/16
Epoch 24 loss 11.5022967926059 valid acc 2/16
Epoch 24 loss 11.513564553871113 valid acc 2/16
Epoch 24 loss 11.599121872028524 valid acc 2/16
Epoch 24 loss 11.460009179902 valid acc 2/16
Epoch 24 loss 11.470465610069983 valid acc 2/16
Epoch 24 loss 11.522935626161797 valid acc 2/16
Epoch 24 loss 11.497357535839388 valid acc 2/16
Epoch 24 loss 11.530336748503924 valid acc 2/16
Epoch 24 loss 11.46003442951854 valid acc 2/16
Epoch 24 loss 11.478317754660427 valid acc 2/16
Epoch 24 loss 11.515248578438978 valid acc 2/16
Epoch 24 loss 11.481460165600472 valid acc 2/16
Epoch 24 loss 11.490291598887106 valid acc 2/16
Epoch 24 loss 11.445307674091428 valid acc 2/16
Epoch 24 loss 11.564334566306146 valid acc 2/16
Epoch 24 loss 11.561626701369025 valid acc 2/16
Epoch 24 loss 11.418847114014312 valid acc 2/16
Epoch 24 loss 11.508448132300119 valid acc 2/16
Epoch 24 loss 11.448324839381245 valid acc 2/16
Epoch 25 loss 2.283980041455281 valid acc 2/16
Epoch 25 loss 11.471479686280038 valid acc 2/16
Epoch 25 loss 11.466286267898345 valid acc 2/16
Epoch 25 loss 11.507706454555201 valid acc 2/16
Epoch 25 loss 11.541235575797636 valid acc 2/16
Epoch 25 loss 11.415606965877949 valid acc 2/16
Epoch 25 loss 11.495493476521776 valid acc 2/16
Epoch 25 loss 11.55550978733253 valid acc 2/16
Epoch 25 loss 11.486678547539924 valid acc 2/16
Epoch 25 loss 11.510417902648747 valid acc 2/16
Epoch 25 loss 11.577257827136485 valid acc 2/16
Epoch 25 loss 11.491282555160186 valid acc 2/16
Epoch 25 loss 11.485867721759876 valid acc 2/16
Epoch 25 loss 11.44413727215383 valid acc 2/16
Epoch 25 loss 11.474605411308918 valid acc 2/16
Epoch 25 loss 11.434005431053329 valid acc 2/16
Epoch 25 loss 11.518119230605592 valid acc 2/16
Epoch 25 loss 11.527968087054925 valid acc 2/16
Epoch 25 loss 11.525733411623253 valid acc 2/16
Epoch 25 loss 11.54415936810637 valid acc 2/16
Epoch 25 loss 11.514959095950825 valid acc 2/16
Epoch 25 loss 11.463209099226896 valid acc 2/16
Epoch 25 loss 11.483250880627853 valid acc 2/16
Epoch 25 loss 11.511990564977111 valid acc 2/16
Epoch 25 loss 11.530167934546641 valid acc 2/16
Epoch 25 loss 11.543942474935267 valid acc 2/16
Epoch 25 loss 11.517242583293303 valid acc 2/16
Epoch 25 loss 11.481232216186294 valid acc 2/16
Epoch 25 loss 11.47802270165352 valid acc 2/16
Epoch 25 loss 11.574558866674273 valid acc 2/16
Epoch 25 loss 11.523236920548054 valid acc 2/16
Epoch 25 loss 11.427205078746052 valid acc 2/16
Epoch 25 loss 11.532688591292825 valid acc 2/16
Epoch 25 loss 11.526534994298952 valid acc 2/16
Epoch 25 loss 11.474481790519 valid acc 2/16
Epoch 25 loss 11.482925714413934 valid acc 2/16
Epoch 25 loss 11.445467880135089 valid acc 2/16
Epoch 25 loss 11.575894853722675 valid acc 2/16
Epoch 25 loss 11.525654672991235 valid acc 2/16
Epoch 25 loss 11.463713874350212 valid acc 2/16
Epoch 25 loss 11.557557397637526 valid acc 2/16
Epoch 25 loss 11.478165771356107 valid acc 2/16
Epoch 25 loss 11.490655677882263 valid acc 2/16
Epoch 25 loss 11.497049149309674 valid acc 2/16
Epoch 25 loss 11.50231921331586 valid acc 2/16
Epoch 25 loss 11.513575063028128 valid acc 2/16
Epoch 25 loss 11.59920191483371 valid acc 2/16
Epoch 25 loss 11.459986943859139 valid acc 2/16
Epoch 25 loss 11.470417327896701 valid acc 2/16
Epoch 25 loss 11.522969933170446 valid acc 2/16
Epoch 25 loss 11.497374354740789 valid acc 2/16
Epoch 25 loss 11.53037549873291 valid acc 2/16
Epoch 25 loss 11.460026139216879 valid acc 2/16
Epoch 25 loss 11.478258018804324 valid acc 2/16
Epoch 25 loss 11.515243171975426 valid acc 2/16
Epoch 25 loss 11.481433188129039 valid acc 2/16
Epoch 25 loss 11.490297154292463 valid acc 2/16
Epoch 25 loss 11.445243064145977 valid acc 2/16
Epoch 25 loss 11.564380002220098 valid acc 2/16
Epoch 25 loss 11.561688449265834 valid acc 2/16
Epoch 25 loss 11.418783890327589 valid acc 2/16
Epoch 25 loss 11.508440171965397 valid acc 2/16
Epoch 25 loss 11.448298281152594 valid acc 2/16
Epoch 26 loss 2.2839669354590155 valid acc 2/16
Epoch 26 loss 11.471446399565286 valid acc 2/16
Epoch 26 loss 11.466263401269016 valid acc 2/16
Epoch 26 loss 11.507684556584984 valid acc 2/16
Epoch 26 loss 11.541268269162465 valid acc 2/16
Epoch 26 loss 11.415582048524918 valid acc 2/16
Epoch 26 loss 11.495481690775724 valid acc 2/16
Epoch 26 loss 11.555572375952336 valid acc 2/16
Epoch 26 loss 11.486670661000913 valid acc 2/16
Epoch 26 loss 11.510416648060474 valid acc 2/16
Epoch 26 loss 11.577341670159974 valid acc 2/16
Epoch 26 loss 11.491254321679822 valid acc 2/16
Epoch 26 loss 11.485861710790626 valid acc 2/16
Epoch 26 loss 11.44408966099046 valid acc 2/16
Epoch 26 loss 11.474606386379413 valid acc 2/16
Epoch 26 loss 11.433952555620145 valid acc 2/16
Epoch 26 loss 11.51816827818649 valid acc 2/16
Epoch 26 loss 11.527958468959556 valid acc 2/16
Epoch 26 loss 11.525747807120432 valid acc 2/16
Epoch 26 loss 11.544195341418844 valid acc 2/16
Epoch 26 loss 11.514946998319859 valid acc 2/16
Epoch 26 loss 11.46318088320319 valid acc 2/16
Epoch 26 loss 11.483238008414682 valid acc 2/16
Epoch 26 loss 11.51200949098746 valid acc 2/16
Epoch 26 loss 11.530158620638964 valid acc 2/16
Epoch 26 loss 11.543982139585607 valid acc 2/16
Epoch 26 loss 11.517252238899056 valid acc 2/16
Epoch 26 loss 11.481218777880466 valid acc 2/16
Epoch 26 loss 11.477999307479367 valid acc 2/16
Epoch 26 loss 11.57462166962537 valid acc 2/16
Epoch 26 loss 11.52327217369802 valid acc 2/16
Epoch 26 loss 11.427161596938449 valid acc 2/16
Epoch 26 loss 11.532679099097237 valid acc 2/16
Epoch 26 loss 11.526547183532283 valid acc 2/16
Epoch 26 loss 11.474473291501148 valid acc 2/16
Epoch 26 loss 11.482907497847624 valid acc 2/16
Epoch 26 loss 11.445425401083524 valid acc 2/16
Epoch 26 loss 11.575939987920137 valid acc 2/16
Epoch 26 loss 11.525673959476459 valid acc 2/16
Epoch 26 loss 11.463715092825536 valid acc 2/16
Epoch 26 loss 11.557596059151583 valid acc 2/16
Epoch 26 loss 11.478155190877484 valid acc 2/16
Epoch 26 loss 11.490641193028257 valid acc 2/16
Epoch 26 loss 11.49703560708897 valid acc 2/16
Epoch 26 loss 11.502336168103987 valid acc 2/16
Epoch 26 loss 11.513582341737443 valid acc 2/16
Epoch 26 loss 11.59926105182629 valid acc 2/16
Epoch 26 loss 11.459969910322533 valid acc 2/16
Epoch 26 loss 11.470381404521042 valid acc 2/16
Epoch 26 loss 11.52299572408478 valid acc 2/16
Epoch 26 loss 11.497386806535294 valid acc 2/16
Epoch 26 loss 11.530403872408188 valid acc 2/16
Epoch 26 loss 11.460020452995515 valid acc 2/16
Epoch 26 loss 11.478213335645616 valid acc 2/16
Epoch 26 loss 11.515240011014853 valid acc 2/16
Epoch 26 loss 11.481413092998338 valid acc 2/16
Epoch 26 loss 11.490301593515664 valid acc 2/16
Epoch 26 loss 11.445195258904736 valid acc 2/16
Epoch 26 loss 11.564413427700137 valid acc 2/16
Epoch 26 loss 11.561733827984364 valid acc 2/16
Epoch 26 loss 11.41873745080032 valid acc 2/16
Epoch 26 loss 11.508433911630771 valid acc 2/16
Epoch 26 loss 11.448279244378089 valid acc 2/16
Epoch 27 loss 2.2839573866468044 valid acc 2/16
Epoch 27 loss 11.471420914784582 valid acc 2/16
Epoch 27 loss 11.466246487366679 valid acc 2/16
Epoch 27 loss 11.50766799584151 valid acc 2/16
Epoch 27 loss 11.5412925911563 valid acc 2/16
Epoch 27 loss 11.415563872589743 valid acc 2/16
Epoch 27 loss 11.495472638317372 valid acc 2/16
Epoch 27 loss 11.555619762954898 valid acc 2/16
Epoch 27 loss 11.48666554069046 valid acc 2/16
Epoch 27 loss 11.510415109412921 valid acc 2/16
Epoch 27 loss 11.577403888335672 valid acc 2/16
Epoch 27 loss 11.491233436005016 valid acc 2/16
Epoch 27 loss 11.48585891553622 valid acc 2/16
Epoch 27 loss 11.444054515796747 valid acc 2/16
Epoch 27 loss 11.47460780294814 valid acc 2/16
Epoch 27 loss 11.433912824387939 valid acc 2/16
Epoch 27 loss 11.518204986524148 valid acc 2/16
Epoch 27 loss 11.527951549994617 valid acc 2/16
Epoch 27 loss 11.525758465356954 valid acc 2/16
Epoch 27 loss 11.544222241703768 valid acc 2/16
Epoch 27 loss 11.514937110752092 valid acc 2/16
Epoch 27 loss 11.46315985999177 valid acc 2/16
Epoch 27 loss 11.483227513540674 valid acc 2/16
Epoch 27 loss 11.512024014061794 valid acc 2/16
Epoch 27 loss 11.530150524452301 valid acc 2/16
Epoch 27 loss 11.544011820940797 valid acc 2/16
Epoch 27 loss 11.517259566769633 valid acc 2/16
Epoch 27 loss 11.481209075659946 valid acc 2/16
Epoch 27 loss 11.477981955230202 valid acc 2/16
Epoch 27 loss 11.574668298702173 valid acc 2/16
Epoch 27 loss 11.52329869708589 valid acc 2/16
Epoch 27 loss 11.42712943305498 valid acc 2/16
Epoch 27 loss 11.532671789986757 valid acc 2/16
Epoch 27 loss 11.526556138121125 valid acc 2/16
Epoch 27 loss 11.474467209219222 valid acc 2/16
Epoch 27 loss 11.482893689382255 valid acc 2/16
Epoch 27 loss 11.445393723816094 valid acc 2/16
Epoch 27 loss 11.575973720849072 valid acc 2/16
Epoch 27 loss 11.525687981276764 valid acc 2/16
Epoch 27 loss 11.463715720573543 valid acc 2/16
Epoch 27 loss 11.55762474968447 valid acc 2/16
Epoch 27 loss 11.478147936703772 valid acc 2/16
Epoch 27 loss 11.49062950404927 valid acc 2/16
Epoch 27 loss 11.497025708782523 valid acc 2/16
Epoch 27 loss 11.502348969838614 valid acc 2/16
Epoch 27 loss 11.513587364539402 valid acc 2/16
Epoch 27 loss 11.599304766133313 valid acc 2/16
Epoch 27 loss 11.459956864954542 valid acc 2/16
Epoch 27 loss 11.470354652713022 valid acc 2/16
Epoch 27 loss 11.523015101525962 valid acc 2/16
Epoch 27 loss 11.497396026118054 valid acc 2/16
Epoch 27 loss 11.530424648621363 valid acc 2/16
Epoch 27 loss 11.460016575579981 valid acc 2/16
Epoch 27 loss 11.478179906918129 valid acc 2/16
Epoch 27 loss 11.515238278043173 valid acc 2/16
Epoch 27 loss 11.481398108233229 valid acc 2/16
Epoch 27 loss 11.490305121201665 valid acc 2/16
Epoch 27 loss 11.445159867522072 valid acc 2/16
Epoch 27 loss 11.564438032398163 valid acc 2/16
Epoch 27 loss 11.561767184325056 valid acc 2/16
Epoch 27 loss 11.418703325721598 valid acc 2/16
Epoch 27 loss 11.508429001287057 valid acc 2/16
Epoch 27 loss 11.44826559970903 valid acc 2/16
Epoch 28 loss 2.283950431086363 valid acc 2/16
Epoch 28 loss 11.471401414990336 valid acc 2/16
Epoch 28 loss 11.466233967532418 valid acc 2/16
Epoch 28 loss 11.50765547028871 valid acc 2/16
Epoch 28 loss 11.541310703437713 valid acc 2/16
Epoch 28 loss 11.415550597577987 valid acc 2/16
Epoch 28 loss 11.49546568692474 valid acc 2/16
Epoch 28 loss 11.555655628672149 valid acc 2/16
Epoch 28 loss 11.48666227231304 valid acc 2/16
Epoch 28 loss 11.510413519177389 valid acc 2/16
Epoch 28 loss 11.577450075393132 valid acc 2/16
Epoch 28 loss 11.491217979454191 valid acc 2/16
Epoch 28 loss 11.485858054249483 valid acc 2/16
Epoch 28 loss 11.444028550604049 valid acc 2/16
Epoch 28 loss 11.474609355447217 valid acc 2/16
Epoch 28 loss 11.433882966923779 valid acc 2/16
Epoch 28 loss 11.518232467112023 valid acc 2/16
Epoch 28 loss 11.527946569321312 valid acc 2/16
Epoch 28 loss 11.525766358400805 valid acc 2/16
Epoch 28 loss 11.544242370351432 valid acc 2/16
Epoch 28 loss 11.514929111170998 valid acc 2/16
Epoch 28 loss 11.463144192599266 valid acc 2/16
Epoch 28 loss 11.483219023951113 valid acc 2/16
Epoch 28 loss 11.5120351476614 valid acc 2/16
Epoch 28 loss 11.530143649580822 valid acc 2/16
Epoch 28 loss 11.544034035678633 valid acc 2/16
Epoch 28 loss 11.517265124157117 valid acc 2/16
Epoch 28 loss 11.48120206528908 valid acc 2/16
Epoch 28 loss 11.477969089410832 valid acc 2/16
Epoch 28 loss 11.574702949096256 valid acc 2/16
Epoch 28 loss 11.523318643848516 valid acc 2/16
Epoch 28 loss 11.427105628763295 valid acc 2/16
Epoch 28 loss 11.53266617893757 valid acc 2/16
Epoch 28 loss 11.526562717064294 valid acc 2/16
Epoch 28 loss 11.4744628479314 valid acc 2/16
Epoch 28 loss 11.482883226375206 valid acc 2/16
Epoch 28 loss 11.44537009372402 valid acc 2/16
Epoch 28 loss 11.575998947066914 valid acc 2/16
Epoch 28 loss 11.525698169154774 valid acc 2/16
Epoch 28 loss 11.463715980511598 valid acc 2/16
Epoch 28 loss 11.55764604423466 valid acc 2/16
Epoch 28 loss 11.478142982071187 valid acc 2/16
Epoch 28 loss 11.490620137104058 valid acc 2/16
Epoch 28 loss 11.497018475541825 valid acc 2/16
Epoch 28 loss 11.502358622277125 valid acc 2/16
Epoch 28 loss 11.513590816724555 valid acc 2/16
Epoch 28 loss 11.59933709710807 valid acc 2/16
Epoch 28 loss 11.459946876680606 valid acc 2/16
Epoch 28 loss 11.470334714323691 valid acc 2/16
Epoch 28 loss 11.523029652934747 valid acc 2/16
Epoch 28 loss 11.497402853717716 valid acc 2/16
Epoch 28 loss 11.53043986252213 valid acc 2/16
Epoch 28 loss 11.460013950279636 valid acc 2/16
Epoch 28 loss 11.47815489472876 valid acc 2/16
Epoch 28 loss 11.515237433764703 valid acc 2/16
Epoch 28 loss 11.481386922908095 valid acc 2/16
Epoch 28 loss 11.49030791166493 valid acc 2/16
Epoch 28 loss 11.445133652387065 valid acc 2/16
Epoch 28 loss 11.564456155436309 valid acc 2/16
Epoch 28 loss 11.56179170933072 valid acc 2/16
Epoch 28 loss 11.418678239724677 valid acc 2/16
Epoch 28 loss 11.508425159362169 valid acc 2/16
Epoch 28 loss 11.448255821143345 valid acc 2/16
Epoch 29 loss 2.283945365857741 valid acc 2/16
Epoch 29 loss 11.471386503508926 valid acc 2/16
Epoch 29 loss 11.466224694131652 valid acc 2/16
Epoch 29 loss 11.507645996492343 valid acc 2/16
Epoch 29 loss 11.541324205108817 valid acc 2/16
Epoch 29 loss 11.415540889928007 valid acc 2/16
Epoch 29 loss 11.49546035089988 valid acc 2/16
Epoch 29 loss 11.555682766423999 valid acc 2/16
Epoch 29 loss 11.486660233235803 valid acc 2/16
Epoch 29 loss 11.510412008272727 valid acc 2/16
Epoch 29 loss 11.57748437414519 valid acc 2/16
Epoch 29 loss 11.491206536415277 valid acc 2/16
Epoch 29 loss 11.485858299675918 valid acc 2/16
Epoch 29 loss 11.444009351803235 valid acc 2/16
Epoch 29 loss 11.474610870747716 valid acc 2/16
Epoch 29 loss 11.43386052762978 valid acc 2/16
Epoch 29 loss 11.518253045122066 valid acc 2/16
Epoch 29 loss 11.527942981915896 valid acc 2/16
Epoch 29 loss 11.525772205435675 valid acc 2/16
Epoch 29 loss 11.544257441836455 valid acc 2/16
Epoch 29 loss 11.514922693178287 valid acc 2/16
Epoch 29 loss 11.46313251449045 valid acc 2/16
Epoch 29 loss 11.48321220200333 valid acc 2/16
Epoch 29 loss 11.512043675492611 valid acc 2/16
Epoch 29 loss 11.530137913544742 valid acc 2/16
Epoch 29 loss 11.54405066543827 valid acc 2/16
Epoch 29 loss 11.517269336298646 valid acc 2/16
Epoch 29 loss 11.481196996049903 valid acc 2/16
Epoch 29 loss 11.477959553864736 valid acc 2/16
Epoch 29 loss 11.574728720306203 valid acc 2/16
Epoch 29 loss 11.52333363892745 valid acc 2/16
Epoch 29 loss 11.427088002491216 valid acc 2/16
Epoch 29 loss 11.532661883704371 valid acc 2/16
Epoch 29 loss 11.526567551504039 valid acc 2/16
Epoch 29 loss 11.474459714307388 valid acc 2/16
Epoch 29 loss 11.48287530152901 valid acc 2/16
Epoch 29 loss 11.445352461171254 valid acc 2/16
Epoch 29 loss 11.576017822642882 valid acc 2/16
Epoch 29 loss 11.525705566703811 valid acc 2/16
Epoch 29 loss 11.463716020211233 valid acc 2/16
Epoch 29 loss 11.557661852159075 valid acc 2/16
Epoch 29 loss 11.478139613051646 valid acc 2/16
Epoch 29 loss 11.490612675539564 valid acc 2/16
Epoch 29 loss 11.497013191435835 valid acc 2/16
Epoch 29 loss 11.502365890759952 valid acc 2/16
Epoch 29 loss 11.513593178836931 valid acc 2/16
Epoch 29 loss 11.599361021796952 valid acc 2/16
Epoch 29 loss 11.459939231358128 valid acc 2/16
Epoch 29 loss 11.470319842418794 valid acc 2/16
Epoch 29 loss 11.52304057537306 valid acc 2/16
Epoch 29 loss 11.497407911088597 valid acc 2/16
Epoch 29 loss 11.530451004146252 valid acc 2/16
Epoch 29 loss 11.46001218811787 valid acc 2/16
Epoch 29 loss 11.478136177935415 valid acc 2/16
Epoch 29 loss 11.515237126084177 valid acc 2/16
Epoch 29 loss 11.481378565684706 valid acc 2/16
Epoch 29 loss 11.49031011041177 valid acc 2/16
Epoch 29 loss 11.445114224376162 valid acc 2/16
Epoch 29 loss 11.564469512893071 valid acc 2/16
Epoch 29 loss 11.56180974576284 valid acc 2/16
Epoch 29 loss 11.418659791395998 valid acc 2/16
Epoch 29 loss 11.508422160271873 valid acc 2/16
Epoch 29 loss 11.448248814674326 valid acc 2/16
Epoch 30 loss 2.2839416783550144 valid acc 2/16
Epoch 30 loss 11.471375107298126 valid acc 2/16
Epoch 30 loss 11.46621782116961 valid acc 2/16
Epoch 30 loss 11.507638831112082 valid acc 2/16
Epoch 30 loss 11.541334280059143 valid acc 2/16
Epoch 30 loss 11.415533782121772 valid acc 2/16
Epoch 30 loss 11.495456256703358 valid acc 2/16
Epoch 30 loss 11.555703294918505 valid acc 2/16
Epoch 30 loss 11.48665900147258 valid acc 2/16
Epoch 30 loss 11.510410642997051 valid acc 2/16
Epoch 30 loss 11.577509853728245 valid acc 2/16
Epoch 30 loss 11.491198061721654 valid acc 2/16
Epoch 30 loss 11.485859127577385 valid acc 2/16
Epoch 30 loss 11.44399514474446 valid acc 2/16
Epoch 30 loss 11.474612258782471 valid acc 2/16
Epoch 30 loss 11.433843662238516 valid acc 2/16
Epoch 30 loss 11.51826845845387 valid acc 2/16
Epoch 30 loss 11.527940396828537 valid acc 2/16
Epoch 30 loss 11.525776538371833 valid acc 2/16
Epoch 30 loss 11.544268734073713 valid acc 2/16
Epoch 30 loss 11.514917580198711 valid acc 2/16
Epoch 30 loss 11.463123808565566 valid acc 2/16
Epoch 30 loss 11.483206751173883 valid acc 2/16
Epoch 30 loss 11.51205020238094 valid acc 2/16
Epoch 30 loss 11.530133192710016 valid acc 2/16
Epoch 30 loss 11.54406311703713 valid acc 2/16
Epoch 30 loss 11.517272527148204 valid acc 2/16
Epoch 30 loss 11.481193327697087 valid acc 2/16
Epoch 30 loss 11.477952489431148 valid acc 2/16
Epoch 30 loss 11.57474790411993 valid acc 2/16
Epoch 30 loss 11.523344907654279 valid acc 2/16
Epoch 30 loss 11.42707494442254 valid acc 2/16
Epoch 30 loss 11.532658604478137 valid acc 2/16
Epoch 30 loss 11.526571104920183 valid acc 2/16
Epoch 30 loss 11.474457457973417 valid acc 2/16
Epoch 30 loss 11.48286930160119 valid acc 2/16
Epoch 30 loss 11.445339300217867 valid acc 2/16
Epoch 30 loss 11.576031954400767 valid acc 2/16
Epoch 30 loss 11.52571093461639 valid acc 2/16
Epoch 30 loss 11.463715935733276 valid acc 2/16
Epoch 30 loss 11.557673589292504 valid acc 2/16
Epoch 30 loss 11.47813733409575 valid acc 2/16
Epoch 30 loss 11.490606762159457 valid acc 2/16
Epoch 30 loss 11.497009332673862 valid acc 2/16
Epoch 30 loss 11.502371357481485 valid acc 2/16
Epoch 30 loss 11.513594786924042 valid acc 2/16
Epoch 30 loss 11.599378735367956 valid acc 2/16
Epoch 30 loss 11.459933381195627 valid acc 2/16
Epoch 30 loss 11.470308741371852 valid acc 2/16
Epoch 30 loss 11.523048770629646 valid acc 2/16
Epoch 30 loss 11.497411658201996 valid acc 2/16
Epoch 30 loss 11.53045916429134 valid acc 2/16
Epoch 30 loss 11.460011017961428 valid acc 2/16
Epoch 30 loss 11.478122170667557 valid acc 2/16
Epoch 30 loss 11.515237128155952 valid acc 2/16
Epoch 30 loss 11.481372315843412 valid acc 2/16
Epoch 30 loss 11.490311837142947 valid acc 2/16
Epoch 30 loss 11.445099819246758 valid acc 2/16
Epoch 30 loss 11.564479364363354 valid acc 2/16
Epoch 30 loss 11.561823013828379 valid acc 2/16
Epoch 30 loss 11.41864621925461 valid acc 2/16
Epoch 30 loss 11.508419824043386 valid acc 2/16
Epoch 30 loss 11.448243795807205 valid acc 2/16
loss around 11 ....
```

### avgpool2d
```
Epoch 1 loss 2.309780487346543 valid acc 1/16
Epoch 1 loss 11.54743639059584 valid acc 1/16
Epoch 1 loss 11.504747088432397 valid acc 1/16
Epoch 1 loss 11.532286085426174 valid acc 1/16
Epoch 1 loss 11.5334291119438 valid acc 1/16
Epoch 1 loss 11.477986516080891 valid acc 2/16
Epoch 1 loss 11.52358235727558 valid acc 2/16
Epoch 1 loss 11.509887921729508 valid acc 3/16
Epoch 1 loss 11.520529344412997 valid acc 3/16
Epoch 1 loss 11.505751114058848 valid acc 3/16
Epoch 1 loss 11.51503537299016 valid acc 3/16
Epoch 1 loss 11.500554204577984 valid acc 3/16
Epoch 1 loss 11.495942882454449 valid acc 4/16
Epoch 1 loss 11.495797280254681 valid acc 2/16
Epoch 1 loss 11.500750883830975 valid acc 2/16
Epoch 1 loss 11.493528183383823 valid acc 2/16
Epoch 1 loss 11.48745589743609 valid acc 2/16
Epoch 1 loss 11.521844630725694 valid acc 2/16
Epoch 1 loss 11.503109636248283 valid acc 2/16
Epoch 1 loss 11.518484157013997 valid acc 2/16
Epoch 1 loss 11.515172791651876 valid acc 2/16
Epoch 1 loss 11.49424900879542 valid acc 2/16
Epoch 1 loss 11.508616188134376 valid acc 2/16
Epoch 1 loss 11.500146468370145 valid acc 2/16
Epoch 1 loss 11.520990032659787 valid acc 2/16
Epoch 1 loss 11.508886377101923 valid acc 3/16
Epoch 1 loss 11.487353264219186 valid acc 2/16
Epoch 1 loss 11.501319802683238 valid acc 3/16
Epoch 1 loss 11.479907657760801 valid acc 1/16
Epoch 1 loss 11.508378861434496 valid acc 1/16
Epoch 1 loss 11.492047922459598 valid acc 1/16
Epoch 1 loss 11.468576712710659 valid acc 1/16
Epoch 1 loss 11.513876959544415 valid acc 1/16
Epoch 1 loss 11.509964453420581 valid acc 1/16
Epoch 1 loss 11.47986263424189 valid acc 2/16
Epoch 1 loss 11.486265422044614 valid acc 2/16
Epoch 1 loss 11.477075747295011 valid acc 3/16
Epoch 1 loss 11.495884610953226 valid acc 4/16
Epoch 1 loss 11.466712623202456 valid acc 4/16
Epoch 1 loss 11.462840041023302 valid acc 3/16
Epoch 1 loss 11.46324329105768 valid acc 5/16
Epoch 1 loss 11.441610494378624 valid acc 4/16
Epoch 1 loss 11.449825096521849 valid acc 6/16
Epoch 1 loss 11.445193000634243 valid acc 8/16
Epoch 1 loss 11.378875679068326 valid acc 7/16
Epoch 1 loss 11.396142124678939 valid acc 8/16
Epoch 1 loss 11.253404116720562 valid acc 7/16
Epoch 1 loss 11.201169113894254 valid acc 7/16
Epoch 1 loss 10.997249261043867 valid acc 7/16
Epoch 1 loss 10.889403625987342 valid acc 7/16
Epoch 1 loss 10.83901427549123 valid acc 9/16
Epoch 1 loss 10.100340462281093 valid acc 7/16
Epoch 1 loss 10.233214785978923 valid acc 10/16
Epoch 1 loss 8.5090466833368 valid acc 8/16
Epoch 1 loss 7.85017088716967 valid acc 12/16
Epoch 1 loss 7.359123246336025 valid acc 10/16
Epoch 1 loss 6.087295285887181 valid acc 10/16
Epoch 1 loss 5.481846579080366 valid acc 11/16
Epoch 1 loss 5.227802402284492 valid acc 10/16
Epoch 1 loss 5.946582458483688 valid acc 11/16
Epoch 1 loss 6.0435217605879314 valid acc 13/16
Epoch 1 loss 5.108042098528947 valid acc 14/16
Epoch 1 loss 5.001983718678238 valid acc 13/16
Epoch 2 loss 0.5969831106985541 valid acc 13/16
Epoch 2 loss 5.1466216104613185 valid acc 13/16
Epoch 2 loss 4.72702344316215 valid acc 14/16
Epoch 2 loss 4.215840655087458 valid acc 11/16
Epoch 2 loss 4.194258822867416 valid acc 14/16
Epoch 2 loss 3.7544941447195628 valid acc 14/16
Epoch 2 loss 3.7047861539940863 valid acc 13/16
Epoch 2 loss 4.208511818395554 valid acc 13/16
Epoch 2 loss 4.910299381708823 valid acc 13/16
Epoch 2 loss 3.285340777385567 valid acc 14/16
Epoch 2 loss 3.5668622389243634 valid acc 14/16
Epoch 2 loss 5.059389326504953 valid acc 14/16
Epoch 2 loss 4.085047012225445 valid acc 15/16
Epoch 2 loss 4.31511917840072 valid acc 13/16
Epoch 2 loss 5.071099376205742 valid acc 14/16
Epoch 2 loss 3.2865257246441875 valid acc 13/16
Epoch 2 loss 5.54034243471026 valid acc 12/16
Epoch 2 loss 3.7100068578185192 valid acc 13/16
Epoch 2 loss 4.279566985301917 valid acc 14/16
Epoch 2 loss 3.599154369724652 valid acc 14/16
Epoch 2 loss 4.01329193315968 valid acc 13/16
Epoch 2 loss 3.8823985023186296 valid acc 12/16
Epoch 2 loss 2.3060888036680067 valid acc 16/16
Epoch 2 loss 3.3772703139086473 valid acc 15/16
Epoch 2 loss 3.05607648963489 valid acc 14/16
Epoch 2 loss 3.1276018371826875 valid acc 14/16
Epoch 2 loss 2.6380034926848692 valid acc 14/16
Epoch 2 loss 2.0150286338357932 valid acc 13/16
Epoch 2 loss 2.3090614798020663 valid acc 13/16
Epoch 2 loss 2.5900026686039452 valid acc 15/16
Epoch 2 loss 3.6443225631933336 valid acc 14/16
Epoch 2 loss 3.52546056115677 valid acc 14/16
Epoch 2 loss 2.006894922357131 valid acc 14/16
Epoch 2 loss 2.9369696305559807 valid acc 15/16
Epoch 2 loss 5.7861282122602855 valid acc 13/16
Epoch 2 loss 4.746155869537777 valid acc 13/16
Epoch 2 loss 2.566178879582891 valid acc 12/16
Epoch 2 loss 2.7078880208945133 valid acc 14/16
Epoch 2 loss 3.6857376032498155 valid acc 14/16
Epoch 2 loss 3.1246419598128554 valid acc 14/16
Epoch 2 loss 2.484151515118838 valid acc 14/16
Epoch 2 loss 2.7295892839552205 valid acc 14/16
Epoch 2 loss 3.1852865131368278 valid acc 14/16
Epoch 2 loss 2.655601972157036 valid acc 16/16
Epoch 2 loss 2.4979660462431603 valid acc 14/16
Epoch 2 loss 1.8570004340711912 valid acc 14/16
Epoch 2 loss 3.0958443186850633 valid acc 15/16
Epoch 2 loss 3.183847063542966 valid acc 14/16
Epoch 2 loss 3.2293582615936973 valid acc 15/16
Epoch 2 loss 2.179534930057012 valid acc 15/16
Epoch 2 loss 2.327029496331929 valid acc 16/16
Epoch 2 loss 2.977884303455311 valid acc 14/16
Epoch 2 loss 3.5772325769067455 valid acc 14/16
Epoch 2 loss 2.675673483662695 valid acc 15/16
Epoch 2 loss 3.291942533434791 valid acc 15/16
Epoch 2 loss 1.8742230812973677 valid acc 15/16
Epoch 2 loss 2.5579138638639782 valid acc 14/16
Epoch 2 loss 2.242927504875044 valid acc 13/16
Epoch 2 loss 3.067791739627905 valid acc 13/16
Epoch 2 loss 2.374150535126681 valid acc 12/16
Epoch 2 loss 3.451774312460565 valid acc 13/16
Epoch 2 loss 2.9142184725479616 valid acc 13/16
Epoch 2 loss 3.529400658459598 valid acc 14/16
Epoch 3 loss 0.2732500310065658 valid acc 14/16
Epoch 3 loss 2.685306449196162 valid acc 15/16
Epoch 3 loss 2.3995730456175903 valid acc 15/16
Epoch 3 loss 2.842601707417977 valid acc 14/16
Epoch 3 loss 2.059393447949972 valid acc 15/16
Epoch 3 loss 1.9528696081844124 valid acc 15/16
Epoch 3 loss 2.473100243760338 valid acc 13/16
Epoch 3 loss 3.390986733146283 valid acc 14/16
Epoch 3 loss 3.21096329783262 valid acc 14/16
Epoch 3 loss 1.8496927758857615 valid acc 15/16
Epoch 3 loss 2.510522514815376 valid acc 15/16
Epoch 3 loss 2.9984293491116185 valid acc 15/16
Epoch 3 loss 2.977108368413748 valid acc 15/16
Epoch 3 loss 3.9176261437468587 valid acc 15/16
Epoch 3 loss 2.795408867730492 valid acc 15/16
Epoch 3 loss 2.245378777124943 valid acc 15/16
Epoch 3 loss 3.2112926869142475 valid acc 14/16
Epoch 3 loss 2.9028104801447836 valid acc 14/16
Epoch 3 loss 2.288935022009069 valid acc 15/16
Epoch 3 loss 1.5630245068576047 valid acc 15/16
Epoch 3 loss 2.909807103649841 valid acc 14/16
Epoch 3 loss 3.020281729773396 valid acc 13/16
Epoch 3 loss 2.529704435938565 valid acc 15/16
Epoch 3 loss 2.713747056991422 valid acc 14/16
Epoch 3 loss 1.6915374446211608 valid acc 14/16
Epoch 3 loss 1.6623641745259166 valid acc 15/16
Epoch 3 loss 1.945543956093234 valid acc 15/16
Epoch 3 loss 2.2697197149735353 valid acc 15/16
Epoch 3 loss 2.3887182058469976 valid acc 15/16
Epoch 3 loss 1.730650214986338 valid acc 15/16
Epoch 3 loss 2.021477383772287 valid acc 13/16
Epoch 3 loss 2.4665694895813424 valid acc 14/16
Epoch 3 loss 1.071238198226494 valid acc 15/16
Epoch 3 loss 1.6941438397066773 valid acc 14/16
Epoch 3 loss 2.845221318928285 valid acc 13/16
Epoch 3 loss 2.4010604116814087 valid acc 14/16
Epoch 3 loss 1.8397617316852874 valid acc 13/16
Epoch 3 loss 2.687814983650243 valid acc 14/16
Epoch 3 loss 2.4883449115195044 valid acc 13/16
Epoch 3 loss 2.472058530912382 valid acc 14/16
Epoch 3 loss 1.7255067930675554 valid acc 14/16
Epoch 3 loss 1.957616970233057 valid acc 15/16
Epoch 3 loss 2.48593784547631 valid acc 14/16
Epoch 3 loss 1.6172718508190287 valid acc 15/16
Epoch 3 loss 3.447787897567744 valid acc 15/16
Epoch 3 loss 1.6711037106829485 valid acc 14/16
Epoch 3 loss 2.0396417198952657 valid acc 15/16
Epoch 3 loss 2.555678380175362 valid acc 14/16
Epoch 3 loss 1.9572253765670338 valid acc 14/16
Epoch 3 loss 1.4104329621794085 valid acc 15/16
Epoch 3 loss 1.375649025828954 valid acc 15/16
Epoch 3 loss 2.412720945772571 valid acc 16/16
Epoch 3 loss 1.892966956220092 valid acc 15/16
Epoch 3 loss 1.5599468709461564 valid acc 16/16
Epoch 3 loss 2.296681923491167 valid acc 14/16
Epoch 3 loss 1.277602064985887 valid acc 15/16
Epoch 3 loss 1.02957862248956 valid acc 14/16
Epoch 3 loss 2.4882633540532186 valid acc 14/16
Epoch 3 loss 3.193184012667553 valid acc 14/16
Epoch 3 loss 1.7107037017054354 valid acc 15/16
Epoch 3 loss 2.2758156617144247 valid acc 14/16
Epoch 3 loss 1.8455708799389647 valid acc 14/16
Epoch 3 loss 2.6337863593501636 valid acc 14/16
Epoch 4 loss 0.2302950948567441 valid acc 15/16
Epoch 4 loss 2.3725007465829413 valid acc 13/16
Epoch 4 loss 2.33099006614535 valid acc 14/16
Epoch 4 loss 1.7649552070427579 valid acc 14/16
Epoch 4 loss 1.3575829788572016 valid acc 14/16
Epoch 4 loss 1.05224832135087 valid acc 15/16
Epoch 4 loss 1.4613031870315627 valid acc 13/16
Epoch 4 loss 1.5729149436260084 valid acc 13/16
Epoch 4 loss 1.8732803865128163 valid acc 14/16
Epoch 4 loss 2.5047347074063997 valid acc 15/16
Epoch 4 loss 2.063856881541449 valid acc 15/16
Epoch 4 loss 2.641535051924903 valid acc 15/16
Epoch 4 loss 2.4851702796559247 valid acc 15/16
Epoch 4 loss 3.0618489890098974 valid acc 14/16
Epoch 4 loss 3.0078812714481313 valid acc 15/16
Epoch 4 loss 1.72083321832539 valid acc 14/16
Epoch 4 loss 2.9422571786585 valid acc 14/16
Epoch 4 loss 2.745104072696475 valid acc 14/16
Epoch 4 loss 2.017824557869698 valid acc 15/16
Epoch 4 loss 2.026173890376745 valid acc 15/16
Epoch 4 loss 2.205417742088588 valid acc 15/16
Epoch 4 loss 1.5939222383839011 valid acc 13/16
Epoch 4 loss 1.366344139506305 valid acc 15/16
Epoch 4 loss 1.5260495907007199 valid acc 14/16
Epoch 4 loss 1.2306830229752566 valid acc 15/16
Epoch 4 loss 0.8824433425063456 valid acc 15/16
Epoch 4 loss 1.1795124801451897 valid acc 15/16
Epoch 4 loss 0.7330301449657186 valid acc 15/16
Epoch 4 loss 1.9406113231978965 valid acc 15/16
Epoch 4 loss 1.3824138387905336 valid acc 14/16
Epoch 4 loss 1.3603119386403102 valid acc 14/16
Epoch 4 loss 1.6536000920055436 valid acc 15/16
Epoch 4 loss 1.2093949955140166 valid acc 14/16
Epoch 4 loss 1.6331008738569917 valid acc 14/16
Epoch 4 loss 2.1296883257233548 valid acc 14/16
Epoch 4 loss 2.3103050034790806 valid acc 14/16
Epoch 4 loss 1.6230962586847666 valid acc 13/16
Epoch 4 loss 1.9294690335895404 valid acc 15/16
Epoch 4 loss 2.1807800191161535 valid acc 14/16
Epoch 4 loss 2.060742646663676 valid acc 14/16
Epoch 4 loss 1.2973122190972757 valid acc 15/16
Epoch 4 loss 1.2420182270322127 valid acc 15/16
Epoch 4 loss 1.6650880515044786 valid acc 14/16
Epoch 4 loss 1.3188407028715694 valid acc 15/16
Epoch 4 loss 2.3344484313264107 valid acc 15/16
Epoch 4 loss 1.1851359142498163 valid acc 15/16
Epoch 4 loss 2.3803605048171157 valid acc 14/16
Epoch 4 loss 2.5158696869241854 valid acc 15/16
Epoch 4 loss 1.4277487739499293 valid acc 15/16
Epoch 4 loss 1.1045222362014313 valid acc 14/16
Epoch 4 loss 1.3959109256118483 valid acc 13/16
Epoch 4 loss 1.401701732379983 valid acc 14/16
Epoch 4 loss 1.6115571231369294 valid acc 15/16
Epoch 4 loss 1.7850180804658977 valid acc 15/16
Epoch 4 loss 2.056467100240045 valid acc 13/16
Epoch 4 loss 1.2302962125985455 valid acc 15/16
Epoch 4 loss 0.8551458969987014 valid acc 13/16
Epoch 4 loss 1.3226125239922712 valid acc 13/16
Epoch 4 loss 2.03906287998152 valid acc 13/16
Epoch 4 loss 1.05726447461497 valid acc 13/16
Epoch 4 loss 1.3985069684830251 valid acc 13/16
Epoch 4 loss 1.6845973505315932 valid acc 13/16
Epoch 4 loss 2.133552798085664 valid acc 15/16
Epoch 5 loss 0.08472373103673408 valid acc 15/16
Epoch 5 loss 1.8394094734645825 valid acc 15/16
Epoch 5 loss 1.93787163108111 valid acc 14/16
Epoch 5 loss 1.4281681081636481 valid acc 14/16
Epoch 5 loss 1.722132095576614 valid acc 15/16
Epoch 5 loss 0.974963096552384 valid acc 15/16
Epoch 5 loss 1.6870110317857168 valid acc 14/16
Epoch 5 loss 1.767544353586516 valid acc 15/16
Epoch 5 loss 1.5812972791243474 valid acc 15/16
Epoch 5 loss 1.480837601807706 valid acc 15/16
Epoch 5 loss 1.249727852719189 valid acc 15/16
Epoch 5 loss 1.9263869400414304 valid acc 15/16
Epoch 5 loss 2.1116353900817573 valid acc 14/16
Epoch 5 loss 2.736179919698588 valid acc 14/16
Epoch 5 loss 2.605442121498858 valid acc 14/16
Epoch 5 loss 1.5001330690652217 valid acc 13/16
Epoch 5 loss 1.914549377485399 valid acc 14/16
Epoch 5 loss 1.7169423650519375 valid acc 14/16
Epoch 5 loss 1.4458834424772027 valid acc 14/16
Epoch 5 loss 1.4340837382961713 valid acc 15/16
Epoch 5 loss 2.1031108852897034 valid acc 15/16
Epoch 5 loss 1.3232072732660654 valid acc 15/16
Epoch 5 loss 0.6532928018734343 valid acc 15/16
Epoch 5 loss 1.4354393254311328 valid acc 14/16
Epoch 5 loss 1.1373188344934064 valid acc 14/16
Epoch 5 loss 1.3570908440052005 valid acc 14/16
Epoch 5 loss 1.1377739907107967 valid acc 16/16
Epoch 5 loss 1.0818080167271291 valid acc 15/16
Epoch 5 loss 1.4772691264636784 valid acc 14/16
Epoch 5 loss 0.8500449543117697 valid acc 14/16
Epoch 5 loss 1.2666432408440964 valid acc 15/16
Epoch 5 loss 1.5222126987490272 valid acc 13/16
Epoch 5 loss 0.4508114012539567 valid acc 13/16
Epoch 5 loss 1.409332883705078 valid acc 15/16
Epoch 5 loss 1.505479018616957 valid acc 15/16
Epoch 5 loss 1.4949511696070217 valid acc 15/16
Epoch 5 loss 1.5903858748254835 valid acc 14/16
Epoch 5 loss 1.2292197581299442 valid acc 14/16
Epoch 5 loss 1.7436281712464945 valid acc 15/16
Epoch 5 loss 1.5966128916995865 valid acc 14/16
Epoch 5 loss 1.0039909168868346 valid acc 15/16
Epoch 5 loss 1.481316618472054 valid acc 15/16
Epoch 5 loss 1.2765327715092727 valid acc 15/16
Epoch 5 loss 1.2585474354504635 valid acc 14/16
Epoch 5 loss 1.7714070768557768 valid acc 15/16
Epoch 5 loss 0.8607281070112149 valid acc 16/16
Epoch 5 loss 1.5758238959704935 valid acc 16/16
Epoch 5 loss 1.945123857673679 valid acc 14/16
Epoch 5 loss 1.1634578971453182 valid acc 14/16
Epoch 5 loss 0.5985334783475973 valid acc 15/16
Epoch 5 loss 0.8919413403656895 valid acc 15/16
Epoch 5 loss 0.9030381557085282 valid acc 15/16
Epoch 5 loss 1.4776262255651162 valid acc 14/16
Epoch 5 loss 0.8696023600929601 valid acc 16/16
Epoch 5 loss 1.498706669054594 valid acc 14/16
Epoch 5 loss 0.923818796993532 valid acc 16/16
Epoch 5 loss 0.8420804471736405 valid acc 14/16
Epoch 5 loss 1.5285820565668062 valid acc 14/16
Epoch 5 loss 1.7738602823410519 valid acc 14/16
Epoch 5 loss 1.0601788760143873 valid acc 14/16
Epoch 5 loss 1.4762504555099514 valid acc 14/16
Epoch 5 loss 1.0595714548288406 valid acc 15/16
Epoch 5 loss 1.509462928096362 valid acc 14/16
Epoch 6 loss 0.042550910890600924 valid acc 15/16
Epoch 6 loss 1.4624166060029864 valid acc 15/16
Epoch 6 loss 1.348737185310132 valid acc 15/16
Epoch 6 loss 1.63186703296567 valid acc 14/16
Epoch 6 loss 1.0423032031835116 valid acc 13/16
Epoch 6 loss 0.9301812749776529 valid acc 14/16
Epoch 6 loss 1.357772800834763 valid acc 13/16
Epoch 6 loss 1.050384386611619 valid acc 13/16
Epoch 6 loss 1.3643368814975303 valid acc 14/16
Epoch 6 loss 1.2869170654844173 valid acc 14/16
Epoch 6 loss 1.4183038865276827 valid acc 15/16
Epoch 6 loss 1.6971756818526318 valid acc 15/16
Epoch 6 loss 2.2629314419460442 valid acc 16/16
Epoch 6 loss 1.5059558104813797 valid acc 15/16
Epoch 6 loss 1.8604660237921338 valid acc 16/16
Epoch 6 loss 1.112278019734068 valid acc 13/16
Epoch 6 loss 1.7306094963684617 valid acc 14/16
Epoch 6 loss 1.7486961944900716 valid acc 13/16
Epoch 6 loss 0.9107971374568016 valid acc 15/16
Epoch 6 loss 1.3490991372001224 valid acc 14/16
Epoch 6 loss 2.256859538221494 valid acc 14/16
Epoch 6 loss 1.254901744799282 valid acc 15/16
Epoch 6 loss 0.3675610190057065 valid acc 15/16
Epoch 6 loss 1.128664396485233 valid acc 15/16
Epoch 6 loss 0.7610741761357063 valid acc 15/16
Epoch 6 loss 0.9649010953511952 valid acc 15/16
Epoch 6 loss 0.6086629291139822 valid acc 16/16
Epoch 6 loss 1.3425008535059524 valid acc 15/16
Epoch 6 loss 0.7922141693642996 valid acc 14/16
Epoch 6 loss 0.5452744634914874 valid acc 15/16
Epoch 6 loss 1.163440853322963 valid acc 15/16
Epoch 6 loss 1.1152590440988701 valid acc 15/16
Epoch 6 loss 0.6061449208617017 valid acc 15/16
Epoch 6 loss 0.7810643402162131 valid acc 15/16
Epoch 6 loss 1.3974541240685965 valid acc 15/16
Epoch 6 loss 1.521772466344918 valid acc 15/16
Epoch 6 loss 1.4919541378891876 valid acc 14/16
Epoch 6 loss 1.3332462480089347 valid acc 14/16
Epoch 6 loss 1.7543832898833003 valid acc 14/16
Epoch 6 loss 1.2930883518745082 valid acc 14/16
Epoch 6 loss 0.8336158914940186 valid acc 14/16
Epoch 6 loss 0.9456664999646742 valid acc 15/16
Epoch 6 loss 0.7982821876484072 valid acc 14/16
Epoch 6 loss 0.5957924436280811 valid acc 14/16
Epoch 6 loss 1.1996431710586564 valid acc 15/16
Epoch 6 loss 0.4868625536685415 valid acc 16/16
Epoch 6 loss 1.1716554815786446 valid acc 14/16
Epoch 6 loss 1.5320876219254345 valid acc 13/16
Epoch 6 loss 1.011593795719513 valid acc 15/16
Epoch 6 loss 0.8171265875857792 valid acc 15/16
Epoch 6 loss 0.7130381473822887 valid acc 14/16
Epoch 6 loss 0.7298261749620913 valid acc 15/16
Epoch 6 loss 1.038273277208003 valid acc 15/16
Epoch 6 loss 0.7103894990947222 valid acc 16/16
Epoch 6 loss 0.8055867900858394 valid acc 15/16
Epoch 6 loss 0.8033131246984695 valid acc 15/16
Epoch 6 loss 0.6288034651989529 valid acc 15/16
Epoch 6 loss 1.0492641091655706 valid acc 15/16
Epoch 6 loss 1.8804731014605858 valid acc 15/16
Epoch 6 loss 0.8417894020756507 valid acc 14/16
Epoch 6 loss 1.0077625302277138 valid acc 14/16
Epoch 6 loss 1.1574434593579275 valid acc 14/16
Epoch 6 loss 1.9428900374847067 valid acc 15/16
Epoch 7 loss 0.08783321522949633 valid acc 14/16
Epoch 7 loss 1.1290246573294602 valid acc 14/16
Epoch 7 loss 1.1413290899384678 valid acc 14/16
Epoch 7 loss 0.9827817812833465 valid acc 14/16
Epoch 7 loss 0.49547765727878135 valid acc 14/16
Epoch 7 loss 0.617881487178459 valid acc 15/16
Epoch 7 loss 1.3539786521865276 valid acc 14/16
Epoch 7 loss 0.8767172937240373 valid acc 13/16
Epoch 7 loss 1.0976220125723632 valid acc 15/16
Epoch 7 loss 0.7505237272147409 valid acc 14/16
Epoch 7 loss 1.0881116273822955 valid acc 15/16
Epoch 7 loss 1.0794042708075688 valid acc 16/16
Epoch 7 loss 1.671972497993053 valid acc 15/16
Epoch 7 loss 1.345722858992049 valid acc 15/16
Epoch 7 loss 1.0533724930358823 valid acc 15/16
Epoch 7 loss 0.6955908961256546 valid acc 15/16
Epoch 7 loss 1.7814765553801246 valid acc 15/16
Epoch 7 loss 1.266526730956619 valid acc 15/16
Epoch 7 loss 1.139837645843792 valid acc 15/16
Epoch 7 loss 1.5415129148649114 valid acc 15/16
Epoch 7 loss 1.6542554798958458 valid acc 14/16
Epoch 7 loss 0.5868124185759122 valid acc 15/16
Epoch 7 loss 0.4390225488672915 valid acc 15/16
Epoch 7 loss 0.6044178800150086 valid acc 16/16
Epoch 7 loss 0.728315335317589 valid acc 16/16
Epoch 7 loss 1.228989327569157 valid acc 16/16
Epoch 7 loss 0.42377641180045267 valid acc 16/16
Epoch 7 loss 1.102581902816493 valid acc 15/16
Epoch 7 loss 0.8023770427331347 valid acc 15/16
Epoch 7 loss 0.46582387389976 valid acc 15/16
Epoch 7 loss 0.8753324063633818 valid acc 16/16
Epoch 7 loss 0.8493598992215954 valid acc 16/16
Epoch 7 loss 0.6582840223578006 valid acc 16/16
Epoch 7 loss 0.7183214496580486 valid acc 16/16
Epoch 7 loss 1.2966254385928646 valid acc 15/16
Epoch 7 loss 1.1430142342114815 valid acc 16/16
Epoch 7 loss 0.8102395929986779 valid acc 14/16
Epoch 7 loss 0.7221508453478145 valid acc 15/16
Epoch 7 loss 1.0867043919691577 valid acc 14/16
Epoch 7 loss 1.1678106179229735 valid acc 15/16
Epoch 7 loss 0.7624852366616022 valid acc 15/16
Epoch 7 loss 1.0524115380900823 valid acc 16/16
Epoch 7 loss 0.8189250669580279 valid acc 15/16
Epoch 7 loss 0.6397936495268343 valid acc 15/16
Epoch 7 loss 1.199670965082738 valid acc 14/16
Epoch 7 loss 0.45936766968519505 valid acc 15/16
Epoch 7 loss 1.0418355337294298 valid acc 15/16
Epoch 7 loss 1.4159274565708213 valid acc 15/16
Epoch 7 loss 0.7258562968730643 valid acc 15/16
Epoch 7 loss 1.111216830392555 valid acc 14/16
Epoch 7 loss 0.4188929583348574 valid acc 15/16
Epoch 7 loss 0.6320987474720816 valid acc 14/16
Epoch 7 loss 0.8381717938497815 valid acc 15/16
Epoch 7 loss 0.8082491202121559 valid acc 14/16
Epoch 7 loss 0.9065410888386427 valid acc 15/16
Epoch 7 loss 0.8705222805396362 valid acc 16/16
Epoch 7 loss 0.5929395708610603 valid acc 15/16
Epoch 7 loss 0.7366149530729444 valid acc 16/16
Epoch 7 loss 1.0424908129131607 valid acc 16/16
Epoch 7 loss 0.9467094864284731 valid acc 16/16
Epoch 7 loss 0.5641911201712462 valid acc 16/16
Epoch 7 loss 0.7525860254473861 valid acc 15/16
Epoch 7 loss 1.1457031347951943 valid acc 15/16
Epoch 8 loss 0.02902335230643724 valid acc 15/16
Epoch 8 loss 1.3411116883022571 valid acc 15/16
Epoch 8 loss 0.9227522287788329 valid acc 15/16
Epoch 8 loss 0.7490185070446274 valid acc 14/16
Epoch 8 loss 0.8528675766666254 valid acc 14/16
Epoch 8 loss 0.4659502433101129 valid acc 15/16
Epoch 8 loss 1.1221638282445017 valid acc 15/16
Epoch 8 loss 0.6219208511574468 valid acc 15/16
Epoch 8 loss 0.6142786343828164 valid acc 14/16
Epoch 8 loss 0.6222992426332684 valid acc 14/16
Epoch 8 loss 1.2132055176279934 valid acc 15/16
Epoch 8 loss 0.6958381849894896 valid acc 15/16
Epoch 8 loss 1.1442964563578901 valid acc 16/16
Epoch 8 loss 1.1800682260914552 valid acc 16/16
Epoch 8 loss 1.0924022330005119 valid acc 16/16
Epoch 8 loss 0.6136485404158611 valid acc 15/16
Epoch 8 loss 1.8456829556304972 valid acc 16/16
Epoch 8 loss 1.3509475440879661 valid acc 15/16
Epoch 8 loss 0.8416911736721511 valid acc 16/16
Epoch 8 loss 0.9905019383508332 valid acc 16/16
Epoch 8 loss 1.8364571137350212 valid acc 15/16
Epoch 8 loss 0.7751036769616719 valid acc 15/16
Epoch 8 loss 0.3457342604068494 valid acc 15/16
Epoch 8 loss 0.9998023350032152 valid acc 14/16
Epoch 8 loss 0.892625044609992 valid acc 16/16
Epoch 8 loss 1.0644123659927647 valid acc 14/16
Epoch 8 loss 0.7446288780490262 valid acc 16/16
Epoch 8 loss 0.8034946285529747 valid acc 16/16
Epoch 8 loss 0.652226766309181 valid acc 14/16
Epoch 8 loss 0.2749850136286687 valid acc 16/16
Epoch 8 loss 0.7527465739850541 valid acc 15/16
Epoch 8 loss 0.5375067308770526 valid acc 16/16
Epoch 8 loss 0.4114540463797846 valid acc 15/16
Epoch 8 loss 0.7814216065380104 valid acc 15/16
Epoch 8 loss 1.2433827387034633 valid acc 15/16
Epoch 8 loss 1.3938319102806789 valid acc 15/16
Epoch 8 loss 0.8362341305646835 valid acc 15/16
Epoch 8 loss 0.7649617341014713 valid acc 15/16
Epoch 8 loss 1.224988266498182 valid acc 15/16
Epoch 8 loss 0.7158790499853548 valid acc 14/16
Epoch 8 loss 1.080674968268517 valid acc 14/16
Epoch 8 loss 0.5431803130221129 valid acc 16/16
Epoch 8 loss 0.4681175460332277 valid acc 14/16
Epoch 8 loss 0.7281600167596805 valid acc 14/16
Epoch 8 loss 0.7702752809607869 valid acc 14/16
Epoch 8 loss 0.760619604612359 valid acc 14/16
Epoch 8 loss 0.9019939687707448 valid acc 14/16
Epoch 8 loss 1.1976663041370408 valid acc 15/16
Epoch 8 loss 0.8629800844078652 valid acc 16/16
Epoch 8 loss 0.7185503931166795 valid acc 15/16
Epoch 8 loss 0.8161836050729738 valid acc 15/16
Epoch 8 loss 0.6053250233313213 valid acc 15/16
Epoch 8 loss 0.8642827813138276 valid acc 15/16
Epoch 8 loss 0.4540124325310409 valid acc 16/16
Epoch 8 loss 0.7255194366948048 valid acc 14/16
Epoch 8 loss 0.4705523467744683 valid acc 15/16
Epoch 8 loss 0.34752981526388227 valid acc 15/16
Epoch 8 loss 1.0121365013028094 valid acc 15/16
Epoch 8 loss 1.3477356701732175 valid acc 14/16
Epoch 8 loss 0.7380764669534579 valid acc 14/16
Epoch 8 loss 0.33477932612408684 valid acc 14/16
Epoch 8 loss 0.9247732331058617 valid acc 15/16
Epoch 8 loss 0.8446469123657052 valid acc 15/16
Epoch 9 loss 0.02894313138143867 valid acc 15/16
Epoch 9 loss 0.8298506252189708 valid acc 15/16
Epoch 9 loss 0.7651139888462568 valid acc 15/16
Epoch 9 loss 0.7821861913293918 valid acc 15/16
Epoch 9 loss 0.47694044283127035 valid acc 15/16
Epoch 9 loss 0.3742526986988777 valid acc 15/16
Epoch 9 loss 0.8594739719936512 valid acc 15/16
Epoch 9 loss 0.5416330639053248 valid acc 15/16
Epoch 9 loss 0.5582915609110699 valid acc 15/16
Epoch 9 loss 0.5003563883333515 valid acc 14/16
Epoch 9 loss 0.7258867133157276 valid acc 15/16
Epoch 9 loss 1.530864334543321 valid acc 15/16
Epoch 9 loss 1.4182230315436306 valid acc 15/16
Epoch 9 loss 1.1373450756801349 valid acc 14/16
Epoch 9 loss 0.9372004613102838 valid acc 16/16
Epoch 9 loss 0.738766377523921 valid acc 14/16
Epoch 9 loss 0.9707367789078516 valid acc 15/16
Epoch 9 loss 1.0154174105770075 valid acc 16/16
Epoch 9 loss 0.8421603055018957 valid acc 15/16
Epoch 9 loss 1.0522975693718553 valid acc 16/16
Epoch 9 loss 1.3983346789202793 valid acc 15/16
Epoch 9 loss 0.7625080131104852 valid acc 15/16
Epoch 9 loss 0.44078707268316786 valid acc 15/16
Epoch 9 loss 0.5767868294425702 valid acc 16/16
Epoch 9 loss 0.6829522844190485 valid acc 15/16
Epoch 9 loss 0.7152887317856715 valid acc 16/16
Epoch 9 loss 0.5054399431491671 valid acc 16/16
Epoch 9 loss 0.4155277957250861 valid acc 16/16
Epoch 9 loss 0.4733361237883954 valid acc 14/16
Epoch 9 loss 0.41829230927385525 valid acc 16/16
Epoch 9 loss 0.5819679371405957 valid acc 16/16
Epoch 9 loss 0.36023892405411245 valid acc 15/16
Epoch 9 loss 0.3736688671698375 valid acc 15/16
Epoch 9 loss 0.6223428480237265 valid acc 16/16
Epoch 9 loss 1.2884173136136252 valid acc 16/16
Epoch 9 loss 0.7476957805202014 valid acc 16/16
Epoch 9 loss 1.2138590302706491 valid acc 16/16
Epoch 9 loss 0.6263024467134343 valid acc 16/16
Epoch 9 loss 1.2532633008378702 valid acc 15/16
Epoch 9 loss 0.3666718430151693 valid acc 15/16
Epoch 9 loss 0.39317641048344504 valid acc 16/16
Epoch 9 loss 0.7562036952869835 valid acc 16/16
Epoch 9 loss 0.609835109987102 valid acc 15/16
Epoch 9 loss 0.4893758997638706 valid acc 15/16
Epoch 9 loss 1.0368364423544083 valid acc 16/16
Epoch 9 loss 0.23461240740830636 valid acc 16/16
Epoch 9 loss 0.9432264043665812 valid acc 16/16
Epoch 9 loss 0.8267761060878132 valid acc 15/16
Epoch 9 loss 0.3717638131464089 valid acc 16/16
Epoch 9 loss 0.4350883726049813 valid acc 16/16
Epoch 9 loss 0.23114471868424347 valid acc 16/16
Epoch 9 loss 0.5181840212557387 valid acc 16/16
Epoch 9 loss 0.5651896560391992 valid acc 15/16
Epoch 9 loss 0.32921986251821267 valid acc 16/16
Epoch 9 loss 0.7892368241728637 valid acc 16/16
Epoch 9 loss 0.39945538339516745 valid acc 16/16
Epoch 9 loss 0.7444848430931335 valid acc 15/16
Epoch 9 loss 0.5910441097566207 valid acc 16/16
Epoch 9 loss 1.1289747906669327 valid acc 15/16
Epoch 9 loss 0.3241892839540093 valid acc 15/16
Epoch 9 loss 0.2781023866853086 valid acc 15/16
Epoch 9 loss 0.5569420614826737 valid acc 16/16
Epoch 9 loss 1.1039178074381488 valid acc 16/16
Epoch 10 loss 0.030745449388242696 valid acc 16/16
Epoch 10 loss 1.3990520921579546 valid acc 15/16
Epoch 10 loss 0.90691813069027 valid acc 15/16
Epoch 10 loss 0.540680750144255 valid acc 15/16
Epoch 10 loss 0.6414737344304159 valid acc 14/16
Epoch 10 loss 0.47158437738259307 valid acc 14/16
Epoch 10 loss 0.9421418772058654 valid acc 15/16
Epoch 10 loss 0.4489739556823088 valid acc 15/16
Epoch 10 loss 0.6917220312178628 valid acc 15/16
Epoch 10 loss 0.42096679778750196 valid acc 14/16
Epoch 10 loss 0.5504434854206703 valid acc 15/16
Epoch 10 loss 2.141815385582431 valid acc 15/16
Epoch 10 loss 1.096149897343144 valid acc 16/16
Epoch 10 loss 0.9585851290289539 valid acc 16/16
Epoch 10 loss 0.6226017816846537 valid acc 14/16
Epoch 10 loss 0.5582421841162719 valid acc 14/16
Epoch 10 loss 1.0284476355086745 valid acc 14/16
Epoch 10 loss 0.6800630734135911 valid acc 15/16
Epoch 10 loss 0.9044151872165758 valid acc 15/16
Epoch 10 loss 0.7538117834187883 valid acc 15/16
Epoch 10 loss 1.632439772122297 valid acc 14/16
Epoch 10 loss 0.4490636541749154 valid acc 16/16
Epoch 10 loss 0.22644471928303836 valid acc 15/16
Epoch 10 loss 0.7965396966891817 valid acc 16/16
Epoch 10 loss 0.6235519029677559 valid acc 15/16
Epoch 10 loss 0.8360735762671692 valid acc 16/16
Epoch 10 loss 0.4711988598641323 valid acc 16/16
Epoch 10 loss 0.4225778155517137 valid acc 16/16
Epoch 10 loss 0.4571572880280297 valid acc 14/16
Epoch 10 loss 0.28000726692027744 valid acc 15/16
Epoch 10 loss 0.5104695223553133 valid acc 15/16
Epoch 10 loss 0.4036522453883674 valid acc 16/16
Epoch 10 loss 0.10116039898622309 valid acc 16/16
Epoch 10 loss 0.5207056062573305 valid acc 16/16
Epoch 10 loss 1.5090338916512946 valid acc 15/16
Epoch 10 loss 0.49176096039604333 valid acc 15/16
Epoch 10 loss 0.4300784378589728 valid acc 15/16
Epoch 10 loss 0.39491876029891465 valid acc 16/16
Epoch 10 loss 1.3032868932208612 valid acc 15/16
Epoch 10 loss 0.7163951141926299 valid acc 16/16
Epoch 10 loss 0.18381016010049328 valid acc 16/16
Epoch 10 loss 0.884556618896766 valid acc 16/16
Epoch 10 loss 0.4646566234448448 valid acc 15/16
Epoch 10 loss 0.24732014503631478 valid acc 15/16
Epoch 10 loss 0.8457629522889364 valid acc 16/16
Epoch 10 loss 0.27008923396244877 valid acc 16/16
Epoch 10 loss 0.7480515570406616 valid acc 15/16
Epoch 10 loss 0.8902236240493574 valid acc 15/16
Epoch 10 loss 0.29685639793986884 valid acc 16/16
Epoch 10 loss 0.45702177597736415 valid acc 16/16
Epoch 10 loss 0.4928593298191847 valid acc 16/16
Epoch 10 loss 0.5633609821093549 valid acc 16/16
Epoch 10 loss 0.5720013993882969 valid acc 15/16
Epoch 10 loss 0.2609450640710586 valid acc 15/16
Epoch 10 loss 0.632122414067029 valid acc 16/16
Epoch 10 loss 0.3731736773164751 valid acc 16/16
Epoch 10 loss 0.6449454517527611 valid acc 16/16
Epoch 10 loss 0.8904836187497149 valid acc 15/16
Epoch 10 loss 1.02777073745085 valid acc 15/16
Epoch 10 loss 0.7112616869197421 valid acc 16/16
Epoch 10 loss 0.5345099356025795 valid acc 16/16
Epoch 10 loss 0.4155817398949862 valid acc 15/16
Epoch 10 loss 0.8501612396698577 valid acc 16/16
Epoch 11 loss 0.02455431236346689 valid acc 16/16
Epoch 11 loss 0.7736435609207705 valid acc 16/16
Epoch 11 loss 0.5524298911440779 valid acc 15/16
Epoch 11 loss 0.5301339557932827 valid acc 16/16
Epoch 11 loss 0.7315720481344541 valid acc 14/16
Epoch 11 loss 0.33060242507016335 valid acc 16/16
Epoch 11 loss 0.9545990080771285 valid acc 14/16
Epoch 11 loss 0.5708126692292917 valid acc 15/16
Epoch 11 loss 0.7208387429930472 valid acc 15/16
Epoch 11 loss 0.6761110565931636 valid acc 15/16
Epoch 11 loss 0.6931843415841115 valid acc 15/16
Epoch 11 loss 0.5531454678266883 valid acc 16/16
Epoch 11 loss 1.7532248802061732 valid acc 14/16
Epoch 11 loss 1.0034523549319947 valid acc 15/16
Epoch 11 loss 0.8473246474284895 valid acc 15/16
Epoch 11 loss 0.8371321745515794 valid acc 14/16
Epoch 11 loss 1.3490080593475746 valid acc 15/16
Epoch 11 loss 1.9282369981230818 valid acc 15/16
Epoch 11 loss 0.9624259086069932 valid acc 16/16
Epoch 11 loss 0.7734369361626617 valid acc 16/16
Epoch 11 loss 1.3072238976739836 valid acc 15/16
Epoch 11 loss 0.5094241034766781 valid acc 16/16
Epoch 11 loss 0.2580899660037518 valid acc 16/16
Epoch 11 loss 0.6184482323841994 valid acc 16/16
Epoch 11 loss 0.5311809809213315 valid acc 16/16
Epoch 11 loss 0.8433105171725055 valid acc 16/16
Epoch 11 loss 0.41234534783359633 valid acc 16/16
Epoch 11 loss 0.3943592054860268 valid acc 16/16
Epoch 11 loss 0.5985236732302236 valid acc 15/16
Epoch 11 loss 0.2837273193804152 valid acc 16/16
Epoch 11 loss 0.4607058539804011 valid acc 16/16
Epoch 11 loss 0.7472822942188403 valid acc 16/16
Epoch 11 loss 0.4839571896471158 valid acc 16/16
Epoch 11 loss 0.6349776889538502 valid acc 16/16
Epoch 11 loss 0.7957910189494799 valid acc 16/16
Epoch 11 loss 0.24978039650974115 valid acc 16/16
Epoch 11 loss 0.33881111796997926 valid acc 16/16
Epoch 11 loss 0.6872419704346698 valid acc 16/16
Epoch 11 loss 0.6160158107168757 valid acc 16/16
Epoch 11 loss 0.45963316287209666 valid acc 16/16
Epoch 11 loss 0.39978782063569457 valid acc 16/16
Epoch 11 loss 0.6309061873361059 valid acc 16/16
Epoch 11 loss 0.37269917503376404 valid acc 16/16
Epoch 11 loss 0.41102281047105305 valid acc 15/16
Epoch 11 loss 0.7704193170826326 valid acc 16/16
Epoch 11 loss 0.3382769924600268 valid acc 16/16
Epoch 11 loss 0.5686484378311685 valid acc 16/16
Epoch 11 loss 1.243159683287333 valid acc 15/16
Epoch 11 loss 0.27055692296241257 valid acc 15/16
Epoch 11 loss 0.22351768706029823 valid acc 15/16
Epoch 11 loss 0.38934340549799384 valid acc 16/16
Epoch 11 loss 0.5880202279206275 valid acc 16/16
Epoch 11 loss 0.5885026273582736 valid acc 16/16
Epoch 11 loss 0.18732621401847532 valid acc 16/16
Epoch 11 loss 0.6332765352037111 valid acc 16/16
Epoch 11 loss 0.5904015946534753 valid acc 16/16
Epoch 11 loss 0.4168839214955753 valid acc 16/16
Epoch 11 loss 0.39148297852453035 valid acc 16/16
Epoch 11 loss 0.8572005745589194 valid acc 14/16
Epoch 11 loss 0.42800041076315576 valid acc 16/16
Epoch 11 loss 0.07428931706598979 valid acc 16/16
Epoch 11 loss 0.37970458166410676 valid acc 15/16
Epoch 11 loss 0.8484134023672587 valid acc 15/16
Epoch 12 loss 0.0018550742472812631 valid acc 15/16
Epoch 12 loss 0.8258481362213759 valid acc 15/16
Epoch 12 loss 1.1613457775879508 valid acc 15/16
Epoch 12 loss 0.42987246521788924 valid acc 15/16
Epoch 12 loss 0.20017630031189682 valid acc 15/16
Epoch 12 loss 0.448621451723716 valid acc 15/16
Epoch 12 loss 0.5815870191314858 valid acc 15/16
Epoch 12 loss 0.5731467797711993 valid acc 15/16
Epoch 12 loss 0.5115267696139331 valid acc 16/16
Epoch 12 loss 0.2321212098645311 valid acc 16/16
Epoch 12 loss 0.6698075541823236 valid acc 16/16
Epoch 12 loss 0.32555654948223184 valid acc 16/16
Epoch 12 loss 0.9252078389016456 valid acc 16/16
Epoch 12 loss 0.5614435008987817 valid acc 16/16
Epoch 12 loss 0.6739879220101895 valid acc 15/16
Epoch 12 loss 0.3606815184179067 valid acc 14/16
Epoch 12 loss 0.7058644749778351 valid acc 15/16
Epoch 12 loss 0.5646143274332225 valid acc 15/16
Epoch 12 loss 0.38484864429121685 valid acc 16/16
Epoch 12 loss 0.6406651241102814 valid acc 14/16
Epoch 12 loss 1.4420993316039432 valid acc 13/16
Epoch 12 loss 0.776803848179703 valid acc 14/16
Epoch 12 loss 0.25879981231017324 valid acc 15/16
Epoch 12 loss 0.34974021293861285 valid acc 16/16
Epoch 12 loss 0.27509407607112846 valid acc 15/16
Epoch 12 loss 0.926946247094944 valid acc 15/16
Epoch 12 loss 0.48590429880865577 valid acc 16/16
Epoch 12 loss 0.42232352332889117 valid acc 14/16
Epoch 12 loss 0.32322597087344296 valid acc 15/16
Epoch 12 loss 0.17444000813920363 valid acc 15/16
Epoch 12 loss 0.3531076846096617 valid acc 15/16
Epoch 12 loss 0.24145847311031834 valid acc 14/16
Epoch 12 loss 0.30072322698148124 valid acc 15/16
Epoch 12 loss 0.35645037375865707 valid acc 16/16
Epoch 12 loss 0.9474868820131321 valid acc 16/16
Epoch 12 loss 0.4563415926281641 valid acc 15/16
Epoch 12 loss 0.30847592177690125 valid acc 15/16
Epoch 12 loss 0.452876023915123 valid acc 15/16
Epoch 12 loss 1.1611366297336745 valid acc 16/16
Epoch 12 loss 0.639380135555601 valid acc 16/16
Epoch 12 loss 0.3524473724636965 valid acc 16/16
Epoch 12 loss 0.4643865534117285 valid acc 15/16
Epoch 12 loss 0.18445750195732227 valid acc 16/16
Epoch 12 loss 0.859711711780284 valid acc 16/16
Epoch 12 loss 0.912276544312957 valid acc 16/16
Epoch 12 loss 0.1562924615336152 valid acc 16/16
Epoch 12 loss 0.7730788451079615 valid acc 16/16
Epoch 12 loss 0.6890133754794673 valid acc 15/16
Epoch 12 loss 0.5109009267895964 valid acc 16/16
Epoch 12 loss 0.3714917704574534 valid acc 15/16
Epoch 12 loss 0.15345192323346882 valid acc 16/16
Epoch 12 loss 0.26955411108958577 valid acc 16/16
Epoch 12 loss 0.3293814528424566 valid acc 16/16
Epoch 12 loss 0.31729761683266705 valid acc 16/16
Epoch 12 loss 0.4288762030426483 valid acc 16/16
Epoch 12 loss 0.20852991020381417 valid acc 16/16
Epoch 12 loss 0.43046513751556137 valid acc 16/16
Epoch 12 loss 0.3604385554760141 valid acc 16/16
Epoch 12 loss 0.9014709810376114 valid acc 16/16
Epoch 12 loss 0.5709166438616471 valid acc 16/16
Epoch 12 loss 0.19007511024290388 valid acc 16/16
Epoch 12 loss 0.23205396417391994 valid acc 16/16
Epoch 12 loss 0.42275015584699355 valid acc 16/16
Epoch 13 loss 0.04546287195679116 valid acc 16/16
Epoch 13 loss 0.5335299217977368 valid acc 15/16
Epoch 13 loss 0.5113213132930804 valid acc 15/16
Epoch 13 loss 0.4206050516110095 valid acc 15/16
Epoch 13 loss 0.31745005954054634 valid acc 14/16
Epoch 13 loss 0.15071025356498546 valid acc 15/16
Epoch 13 loss 0.7216593526023702 valid acc 14/16
Epoch 13 loss 0.39826471481707876 valid acc 14/16
Epoch 13 loss 0.4205156809894294 valid acc 14/16
Epoch 13 loss 0.7083681507948597 valid acc 15/16
Epoch 13 loss 0.44052373871996997 valid acc 16/16
Epoch 13 loss 0.817415177593654 valid acc 16/16
Epoch 13 loss 0.9848339771904276 valid acc 16/16
Epoch 13 loss 0.5142983634689735 valid acc 16/16
Epoch 13 loss 0.5517719993369883 valid acc 16/16
Epoch 13 loss 0.19140955162857154 valid acc 16/16
Epoch 13 loss 0.7187165331576872 valid acc 15/16
Epoch 13 loss 0.8335189007806252 valid acc 15/16
Epoch 13 loss 0.6890036821593125 valid acc 16/16
Epoch 13 loss 0.5713669696348057 valid acc 16/16
Epoch 13 loss 1.136136898192141 valid acc 16/16
Epoch 13 loss 0.37668817378538505 valid acc 16/16
Epoch 13 loss 0.25032694546236733 valid acc 16/16
Epoch 13 loss 0.21364580132055733 valid acc 16/16
Epoch 13 loss 0.5729489578942055 valid acc 16/16
Epoch 13 loss 0.8566083019806663 valid acc 16/16
Epoch 13 loss 0.26755082457884044 valid acc 16/16
Epoch 13 loss 0.35999208178044756 valid acc 16/16
Epoch 13 loss 0.5065120510445236 valid acc 16/16
Epoch 13 loss 0.18300012876960525 valid acc 16/16
Epoch 13 loss 0.33824307703978085 valid acc 15/16
Epoch 13 loss 0.618674558833745 valid acc 16/16
Epoch 13 loss 0.2106592113003493 valid acc 15/16
Epoch 13 loss 0.5025165095464053 valid acc 15/16
Epoch 13 loss 0.9736755795804266 valid acc 15/16
Epoch 13 loss 0.5737657173549819 valid acc 15/16
Epoch 13 loss 0.3134238810356977 valid acc 15/16
Epoch 13 loss 0.3216419570981813 valid acc 15/16
Epoch 13 loss 0.6182398975772845 valid acc 15/16
Epoch 13 loss 0.39141160645725764 valid acc 15/16
Epoch 13 loss 0.30091075963318903 valid acc 15/16
Epoch 13 loss 0.4885500719058661 valid acc 16/16
Epoch 13 loss 0.17572904973383335 valid acc 16/16
Epoch 13 loss 0.21713475149281367 valid acc 16/16
Epoch 13 loss 0.5384725763901979 valid acc 16/16
Epoch 13 loss 0.05586176792715475 valid acc 16/16
Epoch 13 loss 0.8136313518733098 valid acc 16/16
Epoch 13 loss 1.2467921691683632 valid acc 16/16
Epoch 13 loss 0.2558402515920559 valid acc 16/16
Epoch 13 loss 0.3179948905733235 valid acc 16/16
Epoch 13 loss 0.25578362411348177 valid acc 16/16
Epoch 13 loss 0.24458763812859355 valid acc 16/16
Epoch 13 loss 0.5985930867569003 valid acc 16/16
Epoch 13 loss 0.1500716055329207 valid acc 16/16
Epoch 13 loss 0.575689801113868 valid acc 16/16
Epoch 13 loss 0.22671774553476953 valid acc 16/16
Epoch 13 loss 0.5731773627956454 valid acc 16/16
Epoch 13 loss 0.3361216664830375 valid acc 16/16
Epoch 13 loss 0.44465675773210117 valid acc 16/16
Epoch 13 loss 0.3877307731469281 valid acc 16/16
Epoch 13 loss 0.3893128755145329 valid acc 16/16
Epoch 13 loss 0.2595143729602626 valid acc 16/16
Epoch 13 loss 0.2526307450969332 valid acc 16/16
Epoch 14 loss 0.006628131954791616 valid acc 16/16
Epoch 14 loss 0.7791547210239389 valid acc 16/16
Epoch 14 loss 0.6686029073768641 valid acc 16/16
Epoch 14 loss 0.3675099389811336 valid acc 16/16
Epoch 14 loss 0.5084329044119452 valid acc 16/16
Epoch 14 loss 0.18732241575194408 valid acc 16/16
Epoch 14 loss 0.33166412637799403 valid acc 16/16
Epoch 14 loss 0.2257079742245861 valid acc 15/16
Epoch 14 loss 0.18600136530392486 valid acc 16/16
Epoch 14 loss 0.08391690792796436 valid acc 15/16
Epoch 14 loss 0.4546627868986533 valid acc 15/16
Epoch 14 loss 0.8006354933439346 valid acc 15/16
Epoch 14 loss 0.7433144750423026 valid acc 16/16
Epoch 14 loss 0.46192896182857784 valid acc 16/16
Epoch 14 loss 0.752373120631967 valid acc 16/16
Epoch 14 loss 0.45594384438902746 valid acc 15/16
Epoch 14 loss 0.5874052257409066 valid acc 16/16
Epoch 14 loss 0.514185715212572 valid acc 16/16
Epoch 14 loss 0.28709799927193574 valid acc 16/16
Epoch 14 loss 0.5764029204492058 valid acc 16/16
Epoch 14 loss 1.2400060212778516 valid acc 15/16
Epoch 14 loss 0.2961781218984204 valid acc 16/16
Epoch 14 loss 0.2081167329897945 valid acc 16/16
Epoch 14 loss 0.32164780872339316 valid acc 16/16
Epoch 14 loss 0.18888932954201743 valid acc 16/16
Epoch 14 loss 0.8306185993479562 valid acc 16/16
Epoch 14 loss 0.31446640736785997 valid acc 16/16
Epoch 14 loss 0.21199452796835053 valid acc 16/16
Epoch 14 loss 0.307212579587084 valid acc 15/16
Epoch 14 loss 0.1036441657584859 valid acc 15/16
Epoch 14 loss 0.3403066251561415 valid acc 16/16
Epoch 14 loss 0.2640527476083135 valid acc 16/16
Epoch 14 loss 0.22291891387718593 valid acc 16/16
Epoch 14 loss 0.2276984956392158 valid acc 16/16
Epoch 14 loss 0.8711061273749459 valid acc 16/16
Epoch 14 loss 0.46136324026979875 valid acc 16/16
Epoch 14 loss 0.28245678688808273 valid acc 15/16
Epoch 14 loss 0.1469290214922484 valid acc 15/16
Epoch 14 loss 0.4053721786852905 valid acc 16/16
Epoch 14 loss 0.256157326476053 valid acc 16/16
Epoch 14 loss 0.2866454610534239 valid acc 16/16
Epoch 14 loss 0.6317412520888674 valid acc 15/16
Epoch 14 loss 0.4317642375465463 valid acc 16/16
Epoch 14 loss 0.2403168735123365 valid acc 16/16
Epoch 14 loss 0.8224462624747029 valid acc 16/16
Epoch 14 loss 0.39890881285123836 valid acc 16/16
Epoch 14 loss 0.6433563176767321 valid acc 16/16
Epoch 14 loss 0.4700476428643414 valid acc 16/16
Epoch 14 loss 0.4977335793525875 valid acc 16/16
Epoch 14 loss 0.22111635223711767 valid acc 16/16
Epoch 14 loss 0.25153048459021843 valid acc 16/16
Epoch 14 loss 0.7251863133929305 valid acc 16/16
Epoch 14 loss 0.4968705085619436 valid acc 16/16
Epoch 14 loss 0.20881992401565874 valid acc 16/16
Epoch 14 loss 0.309218559133785 valid acc 16/16
Epoch 14 loss 0.13489858435874905 valid acc 16/16
Epoch 14 loss 0.8048748129807082 valid acc 15/16
Epoch 14 loss 0.30383410877175904 valid acc 16/16
Epoch 14 loss 1.2665858521502413 valid acc 16/16
Epoch 14 loss 0.3428733899960482 valid acc 16/16
Epoch 14 loss 0.4896180037499588 valid acc 16/16
Epoch 14 loss 0.4747230307019909 valid acc 16/16
Epoch 14 loss 0.7229566939579302 valid acc 16/16
Epoch 15 loss 0.022438794179374055 valid acc 16/16
Epoch 15 loss 1.10631822766282 valid acc 16/16
Epoch 15 loss 0.656830273574863 valid acc 16/16
Epoch 15 loss 0.46410031436457605 valid acc 15/16
Epoch 15 loss 0.24748461903108737 valid acc 15/16
Epoch 15 loss 0.2863860162485621 valid acc 16/16
Epoch 15 loss 0.580459887293343 valid acc 16/16
Epoch 15 loss 0.6157186375809218 valid acc 15/16
Epoch 15 loss 0.3386991852630597 valid acc 16/16
Epoch 15 loss 0.3160529337824995 valid acc 15/16
Epoch 15 loss 0.367942578133617 valid acc 16/16
Epoch 15 loss 0.3935864996721956 valid acc 16/16
Epoch 15 loss 0.7271465061390667 valid acc 16/16
Epoch 15 loss 0.32093757681699187 valid acc 16/16
Epoch 15 loss 0.27737017431673155 valid acc 16/16
Epoch 15 loss 0.32072949357136277 valid acc 16/16
Epoch 15 loss 0.624749339438582 valid acc 15/16
Epoch 15 loss 0.5255466544562968 valid acc 15/16
Epoch 15 loss 0.5732330137549387 valid acc 16/16
Epoch 15 loss 0.37219539719892436 valid acc 16/16
Epoch 15 loss 0.8451828483322259 valid acc 16/16
Epoch 15 loss 0.2926063365640623 valid acc 16/16
Epoch 15 loss 0.19237393806237446 valid acc 16/16
Epoch 15 loss 0.3362570550763918 valid acc 16/16
Epoch 15 loss 0.8177929268771478 valid acc 15/16
Epoch 15 loss 0.4467137810581948 valid acc 16/16
Epoch 15 loss 0.32088837056691216 valid acc 16/16
Epoch 15 loss 0.556211458174452 valid acc 16/16
Epoch 15 loss 0.38865763013317917 valid acc 16/16
Epoch 15 loss 0.28498555813780957 valid acc 16/16
Epoch 15 loss 0.5756715790979137 valid acc 15/16
Epoch 15 loss 0.19771229945943886 valid acc 15/16
Epoch 15 loss 0.13410825017155595 valid acc 15/16
Epoch 15 loss 0.30199413818812776 valid acc 15/16
Epoch 15 loss 0.6450632492347683 valid acc 16/16
Epoch 15 loss 0.5383583772414507 valid acc 16/16
Epoch 15 loss 0.34422005718264886 valid acc 16/16
Epoch 15 loss 0.4814546257863875 valid acc 16/16
Epoch 15 loss 0.38022347867348144 valid acc 16/16
Epoch 15 loss 0.1367895591120647 valid acc 16/16
Epoch 15 loss 0.13416876407317913 valid acc 16/16
Epoch 15 loss 0.389903708442337 valid acc 16/16
Epoch 15 loss 0.22069893995683146 valid acc 16/16
Epoch 15 loss 0.28758944285140037 valid acc 16/16
Epoch 15 loss 0.6542810793396683 valid acc 16/16
Epoch 15 loss 0.1717107026645842 valid acc 16/16
Epoch 15 loss 0.3950728468751475 valid acc 16/16
Epoch 15 loss 0.31340950122287176 valid acc 16/16
Epoch 15 loss 0.45461934204140947 valid acc 16/16
Epoch 15 loss 0.15292786160349942 valid acc 15/16
Epoch 15 loss 0.21226222791020397 valid acc 15/16
Epoch 15 loss 0.4917834821342797 valid acc 16/16
Epoch 15 loss 0.9048544898005706 valid acc 15/16
Epoch 15 loss 0.3464168482441058 valid acc 15/16
Epoch 15 loss 0.3893336780035926 valid acc 16/16
Epoch 15 loss 0.24329558392371964 valid acc 16/16
Epoch 15 loss 0.3594309534693635 valid acc 16/16
Epoch 15 loss 0.30906089167765766 valid acc 15/16
Epoch 15 loss 0.9096512911668133 valid acc 16/16
Epoch 15 loss 0.2573380485847441 valid acc 16/16
Epoch 15 loss 0.30977701363106813 valid acc 16/16
Epoch 15 loss 0.4195638758731378 valid acc 15/16
Epoch 15 loss 0.4639852568490065 valid acc 15/16
Epoch 16 loss 0.005440343108653822 valid acc 15/16
Epoch 16 loss 0.8489574444736546 valid acc 15/16
Epoch 16 loss 0.35129021095287083 valid acc 16/16
Epoch 16 loss 0.35973431588163063 valid acc 16/16
Epoch 16 loss 0.24857202537945777 valid acc 16/16
Epoch 16 loss 0.1281828485111861 valid acc 15/16
Epoch 16 loss 0.5695980689963236 valid acc 16/16
Epoch 16 loss 0.1434518876383682 valid acc 16/16
Epoch 16 loss 0.3877930294971543 valid acc 16/16
Epoch 16 loss 0.28346276200642223 valid acc 16/16
Epoch 16 loss 0.2064484083342667 valid acc 16/16
Epoch 16 loss 0.3193249961745472 valid acc 16/16
Epoch 16 loss 0.7884123217220715 valid acc 16/16
Epoch 16 loss 0.3803081735912037 valid acc 16/16
Epoch 16 loss 0.4141659813084243 valid acc 15/16
Epoch 16 loss 0.329616943798913 valid acc 15/16
Epoch 16 loss 0.8435196918032575 valid acc 15/16
Epoch 16 loss 0.41846868371608353 valid acc 16/16
Epoch 16 loss 0.1876109387927818 valid acc 15/16
Epoch 16 loss 0.25418393503961595 valid acc 16/16
Epoch 16 loss 0.977772585728892 valid acc 15/16
Epoch 16 loss 0.6426834551846556 valid acc 16/16
Epoch 16 loss 0.04653010009938674 valid acc 16/16
Epoch 16 loss 0.2382039202027463 valid acc 16/16
Epoch 16 loss 0.10122039358248075 valid acc 16/16
Epoch 16 loss 0.46775854465894806 valid acc 15/16
Epoch 16 loss 0.10903065084613958 valid acc 15/16
Epoch 16 loss 0.1810731590784136 valid acc 15/16
Epoch 16 loss 0.14798698769691265 valid acc 15/16
Epoch 16 loss 0.13183832981545734 valid acc 15/16
Epoch 16 loss 0.26598985990628904 valid acc 15/16
Epoch 16 loss 0.30632438684500324 valid acc 15/16
Epoch 16 loss 0.29231186250034535 valid acc 15/16
Epoch 16 loss 0.20515049721506629 valid acc 15/16
Epoch 16 loss 1.0902737058982925 valid acc 15/16
Epoch 16 loss 0.2037219144611273 valid acc 15/16
Epoch 16 loss 0.1921471022410236 valid acc 16/16
Epoch 16 loss 0.2996715800262818 valid acc 16/16
Epoch 16 loss 0.22800381121310911 valid acc 16/16
Epoch 16 loss 0.4486955103533857 valid acc 16/16
Epoch 16 loss 0.32781104316603954 valid acc 16/16
Epoch 16 loss 0.23876621860087988 valid acc 16/16
Epoch 16 loss 0.10272518204865705 valid acc 16/16
Epoch 16 loss 0.06402038884179967 valid acc 16/16
Epoch 16 loss 0.4854286507074325 valid acc 16/16
Epoch 16 loss 0.2564498426488215 valid acc 16/16
Epoch 16 loss 0.4824059968325727 valid acc 16/16
Epoch 16 loss 0.44343946838101755 valid acc 16/16
Epoch 16 loss 0.2767433898183699 valid acc 16/16
Epoch 16 loss 0.10867618813634056 valid acc 16/16
Epoch 16 loss 0.23825794632373393 valid acc 16/16
Epoch 16 loss 0.2696031572124573 valid acc 16/16
Epoch 16 loss 0.3085308564292869 valid acc 16/16
Epoch 16 loss 0.14832318153918342 valid acc 16/16
Epoch 16 loss 0.3583537902971102 valid acc 16/16
Epoch 16 loss 0.4090463521732399 valid acc 16/16
Epoch 16 loss 0.5476972008145433 valid acc 16/16
Epoch 16 loss 0.6099776770083961 valid acc 16/16
Epoch 16 loss 0.8453252353360783 valid acc 16/16
Epoch 16 loss 0.39862277891022563 valid acc 16/16
Epoch 16 loss 0.17069430695607546 valid acc 16/16
Epoch 16 loss 0.17295441917355042 valid acc 16/16
Epoch 16 loss 0.4185241282920512 valid acc 16/16
Epoch 17 loss 0.01652016703560888 valid acc 16/16
Epoch 17 loss 0.38712354831233037 valid acc 15/16
Epoch 17 loss 0.37967719952159695 valid acc 15/16
Epoch 17 loss 0.22867247831953486 valid acc 15/16
Epoch 17 loss 0.11659569820293791 valid acc 15/16
Epoch 17 loss 0.3890510885671612 valid acc 15/16
Epoch 17 loss 0.3446206355665631 valid acc 15/16
Epoch 17 loss 0.16597054164695768 valid acc 15/16
Epoch 17 loss 0.17142504976150075 valid acc 15/16
Epoch 17 loss 0.48713646363535323 valid acc 15/16
Epoch 17 loss 0.24228064294699825 valid acc 15/16
Epoch 17 loss 0.24387280192613134 valid acc 16/16
Epoch 17 loss 0.8103533650977237 valid acc 16/16
Epoch 17 loss 0.4518442509531497 valid acc 16/16
Epoch 17 loss 0.4354386527456897 valid acc 16/16
Epoch 17 loss 0.3445419104560644 valid acc 16/16
Epoch 17 loss 0.32903419706314163 valid acc 15/16
Epoch 17 loss 0.605054560577419 valid acc 16/16
Epoch 17 loss 0.4287787758393006 valid acc 16/16
Epoch 17 loss 0.6257749130761502 valid acc 15/16
Epoch 17 loss 0.4026103068289159 valid acc 15/16
Epoch 17 loss 0.34536155672440094 valid acc 16/16
Epoch 17 loss 0.1872022553599507 valid acc 16/16
Epoch 17 loss 0.2708819687990859 valid acc 16/16
Epoch 17 loss 0.49822571261666065 valid acc 16/16
Epoch 17 loss 0.5144847755338011 valid acc 16/16
Epoch 17 loss 0.12355724651484251 valid acc 16/16
Epoch 17 loss 0.07997487895795422 valid acc 16/16
Epoch 17 loss 0.2996920532342574 valid acc 16/16
Epoch 17 loss 0.12644273564203845 valid acc 16/16
Epoch 17 loss 0.4950536227034186 valid acc 16/16
Epoch 17 loss 0.10309408187018937 valid acc 16/16
Epoch 17 loss 0.11271032564448419 valid acc 16/16
Epoch 17 loss 0.40946807809901936 valid acc 16/16
Epoch 17 loss 0.6685374915016187 valid acc 16/16
Epoch 17 loss 0.3013781594512893 valid acc 16/16
Epoch 17 loss 0.28738124326859865 valid acc 16/16
Epoch 17 loss 0.5277894366667766 valid acc 16/16
Epoch 17 loss 0.5058357907509279 valid acc 16/16
Epoch 17 loss 0.26367829075866905 valid acc 16/16
Epoch 17 loss 0.0910922497443939 valid acc 16/16
Epoch 17 loss 0.37585210495008003 valid acc 16/16
Epoch 17 loss 0.25243641299103914 valid acc 16/16
Epoch 17 loss 0.23795556188431916 valid acc 16/16
Epoch 17 loss 0.32871115697656356 valid acc 16/16
Epoch 17 loss 0.07808465879976445 valid acc 16/16
Epoch 17 loss 0.19986369064643006 valid acc 16/16
Epoch 17 loss 0.41259693180751206 valid acc 16/16
Epoch 17 loss 0.30858885087476284 valid acc 16/16
Epoch 17 loss 0.34222920325215783 valid acc 16/16
Epoch 17 loss 0.24923189791397204 valid acc 16/16
Epoch 17 loss 0.2772485480771335 valid acc 16/16
Epoch 17 loss 0.7439248484261839 valid acc 16/16
Epoch 17 loss 0.15178587836798157 valid acc 16/16
Epoch 17 loss 0.25557255405319934 valid acc 16/16
Epoch 17 loss 0.13658719369788092 valid acc 16/16
Epoch 17 loss 0.178787667474696 valid acc 16/16
Epoch 17 loss 0.40142236855265306 valid acc 16/16
Epoch 17 loss 0.5989302131642864 valid acc 16/16
Epoch 17 loss 0.7559964757057058 valid acc 16/16
Epoch 17 loss 0.2681151844445127 valid acc 16/16
Epoch 17 loss 0.21171332750094135 valid acc 16/16
Epoch 17 loss 0.5393918502817034 valid acc 16/16
Epoch 18 loss 0.013936506213732713 valid acc 16/16
Epoch 18 loss 0.48918950277427264 valid acc 16/16
Epoch 18 loss 0.5668190362950506 valid acc 16/16
Epoch 18 loss 0.3142034227760589 valid acc 16/16
Epoch 18 loss 0.1398749658488858 valid acc 16/16
Epoch 18 loss 0.09648098820144807 valid acc 16/16
Epoch 18 loss 0.3969319728735051 valid acc 16/16
Epoch 18 loss 0.19674096546707598 valid acc 15/16
Epoch 18 loss 0.2049527230251112 valid acc 16/16
Epoch 18 loss 0.15380450526963163 valid acc 16/16
Epoch 18 loss 0.19582952441866525 valid acc 16/16
Epoch 18 loss 0.23935799233431604 valid acc 16/16
Epoch 18 loss 0.5478778143978855 valid acc 16/16
Epoch 18 loss 0.4432255556677575 valid acc 15/16
Epoch 18 loss 0.7545096297365523 valid acc 16/16
Epoch 18 loss 0.38612839720914727 valid acc 15/16
Epoch 18 loss 0.4009133879539853 valid acc 15/16
Epoch 18 loss 0.31318756705026823 valid acc 15/16
Epoch 18 loss 0.44373736542568587 valid acc 16/16
Epoch 18 loss 0.21085935246216833 valid acc 16/16
Epoch 18 loss 0.8564322739728778 valid acc 15/16
Epoch 18 loss 0.20306901963406554 valid acc 16/16
Epoch 18 loss 0.0888366169247565 valid acc 16/16
Epoch 18 loss 0.10720426258609389 valid acc 16/16
Epoch 18 loss 0.12558573105795334 valid acc 16/16
Epoch 18 loss 0.24300393245861407 valid acc 16/16
Epoch 18 loss 0.15766921861509692 valid acc 16/16
Epoch 18 loss 0.24312309712043312 valid acc 16/16
Epoch 18 loss 0.20584413870673296 valid acc 16/16
Epoch 18 loss 0.11621493356855789 valid acc 16/16
Epoch 18 loss 0.23624094994921424 valid acc 15/16
Epoch 18 loss 0.20215956472104876 valid acc 16/16
Epoch 18 loss 0.21488244809344892 valid acc 16/16
Epoch 18 loss 0.45521041785659205 valid acc 15/16
Epoch 18 loss 0.7723595919394896 valid acc 15/16
Epoch 18 loss 0.40673202747062354 valid acc 15/16
Epoch 18 loss 0.3968130684899408 valid acc 15/16
Epoch 18 loss 0.14236833030050708 valid acc 15/16
Epoch 18 loss 0.6859003944345432 valid acc 15/16
Epoch 18 loss 0.09988889597210077 valid acc 15/16
Epoch 18 loss 0.2201213815879689 valid acc 15/16
Epoch 18 loss 0.7924544607139806 valid acc 15/16
Epoch 18 loss 0.19374172559883138 valid acc 15/16
Epoch 18 loss 0.5761225405406605 valid acc 16/16
Epoch 18 loss 0.4881354392318363 valid acc 16/16
Epoch 18 loss 0.13037601293346118 valid acc 16/16
Epoch 18 loss 0.3957549195608341 valid acc 16/16
Epoch 18 loss 0.515841324874047 valid acc 16/16
Epoch 18 loss 0.20626962235464466 valid acc 16/16
Epoch 18 loss 0.14336696563206652 valid acc 16/16
Epoch 18 loss 0.17672698606296564 valid acc 16/16
Epoch 18 loss 0.2165879324383201 valid acc 16/16
Epoch 18 loss 0.42888393767359584 valid acc 16/16
Epoch 18 loss 0.13907250997156842 valid acc 15/16
Epoch 18 loss 0.2958829688029532 valid acc 16/16
Epoch 18 loss 0.18344955160590076 valid acc 16/16
Epoch 18 loss 0.41447965893989963 valid acc 15/16
Epoch 18 loss 0.17542601227058213 valid acc 15/16
Epoch 18 loss 0.30707148641083526 valid acc 16/16
Epoch 18 loss 0.18101194272216153 valid acc 16/16
Epoch 18 loss 0.24458884008502113 valid acc 16/16
Epoch 18 loss 0.2004125930092402 valid acc 16/16
Epoch 18 loss 0.23457143572205874 valid acc 16/16
Epoch 19 loss 0.01629592944541558 valid acc 16/16
Epoch 19 loss 0.4055516583652915 valid acc 16/16
Epoch 19 loss 0.6543109764275401 valid acc 16/16
Epoch 19 loss 0.09546476674829618 valid acc 16/16
Epoch 19 loss 0.1408887975954366 valid acc 16/16
Epoch 19 loss 0.10286798125989133 valid acc 16/16
Epoch 19 loss 0.23809085196242347 valid acc 15/16
Epoch 19 loss 0.3805162931068451 valid acc 16/16
Epoch 19 loss 0.31969767745362204 valid acc 16/16
Epoch 19 loss 0.08449885219907605 valid acc 16/16
Epoch 19 loss 0.11136465161850556 valid acc 16/16
Epoch 19 loss 0.5011782161942171 valid acc 15/16
Epoch 19 loss 0.6322126562998066 valid acc 15/16
Epoch 19 loss 0.13411684258654094 valid acc 15/16
Epoch 19 loss 0.49301025237362794 valid acc 16/16
Epoch 19 loss 0.15629338690906386 valid acc 16/16
Epoch 19 loss 0.5561974951884368 valid acc 16/16
Epoch 19 loss 0.6818061075549611 valid acc 15/16
Epoch 19 loss 0.351341394957607 valid acc 16/16
Epoch 19 loss 0.18027000450691444 valid acc 16/16
Epoch 19 loss 0.653229720899271 valid acc 16/16
Epoch 19 loss 0.08636744325816553 valid acc 16/16
Epoch 19 loss 0.20098294285010876 valid acc 16/16
Epoch 19 loss 0.40405446634232006 valid acc 16/16
Epoch 19 loss 0.12628226167232798 valid acc 16/16
Epoch 19 loss 0.34488227402065474 valid acc 16/16
Epoch 19 loss 0.266613222981846 valid acc 16/16
Epoch 19 loss 0.1528093750918282 valid acc 16/16
Epoch 19 loss 0.17752116589901618 valid acc 15/16
Epoch 19 loss 0.06757454586507394 valid acc 16/16
Epoch 19 loss 0.39912422322848196 valid acc 16/16
Epoch 19 loss 0.2084135826548575 valid acc 16/16
Epoch 19 loss 0.13960359039146536 valid acc 16/16
Epoch 19 loss 0.14242133597928458 valid acc 16/16
Epoch 19 loss 0.8513769237662644 valid acc 16/16
Epoch 19 loss 0.14592554448722173 valid acc 16/16
Epoch 19 loss 0.26472507861587613 valid acc 15/16
Epoch 19 loss 0.2244748253842453 valid acc 16/16
Epoch 19 loss 0.14221988599019705 valid acc 16/16
Epoch 19 loss 0.06926973381368268 valid acc 16/16
Epoch 19 loss 0.11253127442281097 valid acc 16/16
Epoch 19 loss 0.20053120837841262 valid acc 15/16
Epoch 19 loss 0.11085046441014768 valid acc 16/16
Epoch 19 loss 0.10093871200390014 valid acc 16/16
Epoch 19 loss 0.25634362737698246 valid acc 16/16
Epoch 19 loss 0.09471527667338059 valid acc 16/16
Epoch 19 loss 0.40550719454947476 valid acc 16/16
Epoch 19 loss 0.33830357084456103 valid acc 16/16
Epoch 19 loss 0.2666718238992443 valid acc 16/16
Epoch 19 loss 0.30304395257676564 valid acc 16/16
Epoch 19 loss 0.3186428696475895 valid acc 16/16
Epoch 19 loss 0.36617076157828554 valid acc 15/16
Epoch 19 loss 0.6615413018144711 valid acc 15/16
Epoch 19 loss 0.16787510302044534 valid acc 15/16
Epoch 19 loss 0.12819768434773385 valid acc 16/16
Epoch 19 loss 0.23925373546050246 valid acc 16/16
Epoch 19 loss 0.4458932589033403 valid acc 16/16
Epoch 19 loss 0.1895166747133682 valid acc 16/16
Epoch 19 loss 0.3560715209815051 valid acc 16/16
Epoch 19 loss 0.28377748453213447 valid acc 16/16
Epoch 19 loss 0.13293533895945825 valid acc 16/16
Epoch 19 loss 0.08620316673850392 valid acc 16/16
Epoch 19 loss 0.31014620243776275 valid acc 16/16
Epoch 20 loss 0.0025538258854569396 valid acc 16/16
Epoch 20 loss 0.4113788161378696 valid acc 16/16
Epoch 20 loss 0.3914792776476005 valid acc 16/16
Epoch 20 loss 0.34551532148495523 valid acc 16/16
Epoch 20 loss 0.44436004025180587 valid acc 16/16
Epoch 20 loss 0.08434570605041286 valid acc 16/16
Epoch 20 loss 0.45119164481582263 valid acc 16/16
Epoch 20 loss 0.2681115517826598 valid acc 16/16
Epoch 20 loss 0.2792686908936116 valid acc 16/16
Epoch 20 loss 0.12355779165593073 valid acc 15/16
Epoch 20 loss 0.2678477212281196 valid acc 15/16
Epoch 20 loss 0.07592944509281518 valid acc 16/16
Epoch 20 loss 0.5054949475121764 valid acc 16/16
Epoch 20 loss 0.5524318782615197 valid acc 16/16
Epoch 20 loss 0.4742826675486152 valid acc 15/16
Epoch 20 loss 0.28627280779686426 valid acc 16/16
Epoch 20 loss 0.28428655821301724 valid acc 16/16
Epoch 20 loss 0.2480448800727757 valid acc 15/16
Epoch 20 loss 0.3603499188626393 valid acc 15/16
Epoch 20 loss 0.12924405431080685 valid acc 16/16
Epoch 20 loss 0.8441522461088732 valid acc 16/16
Epoch 20 loss 0.14148021240570008 valid acc 16/16
Epoch 20 loss 0.2104386273368466 valid acc 15/16
Epoch 20 loss 0.25438586563251325 valid acc 16/16
Epoch 20 loss 0.3706913219777838 valid acc 14/16
Epoch 20 loss 0.6235629811156554 valid acc 15/16
Epoch 20 loss 0.269777446897573 valid acc 15/16
Epoch 20 loss 0.28202921034006556 valid acc 15/16
Epoch 20 loss 0.4613387217517153 valid acc 16/16
Epoch 20 loss 0.035561463337731636 valid acc 16/16
Epoch 20 loss 0.1400375349091757 valid acc 16/16
Epoch 20 loss 0.2840394390288884 valid acc 16/16
Epoch 20 loss 0.23470569266028002 valid acc 16/16
Epoch 20 loss 0.15494467105612603 valid acc 16/16
Epoch 20 loss 0.8710070072892722 valid acc 16/16
Epoch 20 loss 0.12722867994586207 valid acc 16/16
Epoch 20 loss 0.2545377309898038 valid acc 16/16
Epoch 20 loss 0.3609134706211601 valid acc 16/16
Epoch 20 loss 0.3197910468705692 valid acc 16/16
Epoch 20 loss 0.11953988263765469 valid acc 16/16
Epoch 20 loss 0.09885359373615482 valid acc 16/16
Epoch 20 loss 0.43186032766953064 valid acc 16/16
Epoch 20 loss 0.14701027146917123 valid acc 16/16
Epoch 20 loss 0.2744880517688388 valid acc 16/16
Epoch 20 loss 0.30508197416833716 valid acc 16/16
Epoch 20 loss 0.11865027614890097 valid acc 16/16
Epoch 20 loss 0.3916626700072037 valid acc 16/16
Epoch 20 loss 0.49356398999815465 valid acc 16/16
Epoch 20 loss 0.16338527209768522 valid acc 16/16
Epoch 20 loss 0.26970337962123364 valid acc 16/16
Epoch 20 loss 0.19239770249807653 valid acc 16/16
Epoch 20 loss 0.2977066793659449 valid acc 16/16
Epoch 20 loss 0.10241029832606885 valid acc 16/16
Epoch 20 loss 0.13438297546364003 valid acc 16/16
Epoch 20 loss 0.22789291042050822 valid acc 16/16
Epoch 20 loss 0.08637749160529706 valid acc 16/16
Epoch 20 loss 0.46534906260319586 valid acc 16/16
Epoch 20 loss 0.14568602197534739 valid acc 16/16
Epoch 20 loss 0.4983423017190462 valid acc 16/16
Epoch 20 loss 0.18493504211523112 valid acc 16/16
Epoch 20 loss 0.26045211805033935 valid acc 16/16
Epoch 20 loss 0.13060220058091587 valid acc 16/16
Epoch 20 loss 0.4012552245530001 valid acc 15/16
Epoch 21 loss 0.004330388153501508 valid acc 16/16
Epoch 21 loss 0.5633551932655935 valid acc 16/16
Epoch 21 loss 0.4885456390942534 valid acc 16/16
Epoch 21 loss 0.25455974640996193 valid acc 16/16
Epoch 21 loss 0.17937777019562945 valid acc 16/16
Epoch 21 loss 0.11891981553804654 valid acc 16/16
Epoch 21 loss 0.2640995590925255 valid acc 16/16
Epoch 21 loss 0.20954658528124556 valid acc 16/16
Epoch 21 loss 0.25254176585936466 valid acc 16/16
Epoch 21 loss 0.14555078853670256 valid acc 15/16
Epoch 21 loss 0.18109617913386022 valid acc 15/16
Epoch 21 loss 0.24758301927592202 valid acc 16/16
Epoch 21 loss 0.3685525993624417 valid acc 16/16
Epoch 21 loss 0.21714811570033238 valid acc 16/16
Epoch 21 loss 0.7682659035418689 valid acc 16/16
Epoch 21 loss 0.23451718396698623 valid acc 16/16
Epoch 21 loss 0.32539867485300744 valid acc 15/16
Epoch 21 loss 0.42433251244021697 valid acc 15/16
Epoch 21 loss 0.4409576540153172 valid acc 16/16
Epoch 21 loss 0.15820897350614113 valid acc 16/16
Epoch 21 loss 0.7929058124552639 valid acc 14/16
Epoch 21 loss 0.06256969008332114 valid acc 15/16
Epoch 21 loss 0.10429341440168721 valid acc 16/16
Epoch 21 loss 0.20515260531347723 valid acc 16/16
Epoch 21 loss 0.3394801841531103 valid acc 16/16
Epoch 21 loss 0.3408618022143174 valid acc 16/16
Epoch 21 loss 0.15763183024693078 valid acc 16/16
Epoch 21 loss 0.2200231794825495 valid acc 16/16
Epoch 21 loss 0.15389246651911015 valid acc 16/16
Epoch 21 loss 0.07373772903803547 valid acc 16/16
Epoch 21 loss 0.08718427544486396 valid acc 15/16
Epoch 21 loss 0.19221565679835634 valid acc 14/16
Epoch 21 loss 0.10273401652040925 valid acc 14/16
Epoch 21 loss 0.4314528366118141 valid acc 14/16
Epoch 21 loss 0.5614062564292246 valid acc 16/16
Epoch 21 loss 0.11812799122008388 valid acc 16/16
Epoch 21 loss 0.14968116689635874 valid acc 15/16
Epoch 21 loss 0.10703316667180623 valid acc 16/16
Epoch 21 loss 0.217491566062609 valid acc 16/16
Epoch 21 loss 0.213212263392703 valid acc 16/16
Epoch 21 loss 0.1749723249893796 valid acc 16/16
Epoch 21 loss 0.08349101820515206 valid acc 16/16
Epoch 21 loss 0.13895319413729376 valid acc 16/16
Epoch 21 loss 0.12479029998253241 valid acc 16/16
Epoch 21 loss 0.26674928424959826 valid acc 16/16
Epoch 21 loss 0.11975664629849336 valid acc 16/16
Epoch 21 loss 0.3551338186229113 valid acc 16/16
Epoch 21 loss 0.062125750025896376 valid acc 16/16
Epoch 21 loss 0.5288676334396534 valid acc 16/16
Epoch 21 loss 0.15133235558579883 valid acc 16/16
Epoch 21 loss 0.08141729257456845 valid acc 16/16
Epoch 21 loss 0.1008774959026077 valid acc 15/16
Epoch 21 loss 0.37506040854231276 valid acc 16/16
Epoch 21 loss 0.17133932801998009 valid acc 16/16
Epoch 21 loss 0.18372833170749805 valid acc 16/16
Epoch 21 loss 0.0542731104418252 valid acc 16/16
Epoch 21 loss 0.6696826454288265 valid acc 16/16
Epoch 21 loss 0.24210898089125565 valid acc 16/16
Epoch 21 loss 0.5592108101705959 valid acc 16/16
Epoch 21 loss 0.2778935359463187 valid acc 16/16
Epoch 21 loss 0.08710856711107579 valid acc 16/16
Epoch 21 loss 0.0904263360038493 valid acc 15/16
Epoch 21 loss 0.469855126056253 valid acc 15/16
Epoch 22 loss 0.006849219525101036 valid acc 15/16
Epoch 22 loss 0.2959949225204032 valid acc 15/16
Epoch 22 loss 0.38061444239032155 valid acc 15/16
Epoch 22 loss 0.2968131162664629 valid acc 16/16
Epoch 22 loss 0.06823465899200237 valid acc 15/16
Epoch 22 loss 0.10876237258529647 valid acc 15/16
Epoch 22 loss 0.23305473842859012 valid acc 15/16
Epoch 22 loss 0.26233185672224935 valid acc 15/16
Epoch 22 loss 0.548783612558277 valid acc 16/16
Epoch 22 loss 0.11048601924593465 valid acc 16/16
Epoch 22 loss 0.19731454600585846 valid acc 16/16
Epoch 22 loss 0.20805058792029785 valid acc 16/16
Epoch 22 loss 0.4958225428676952 valid acc 16/16
Epoch 22 loss 0.5006011290006203 valid acc 16/16
Epoch 22 loss 0.3334470318558489 valid acc 16/16
Epoch 22 loss 0.13189680523476638 valid acc 16/16
Epoch 22 loss 0.20217543545059274 valid acc 16/16
Epoch 22 loss 0.4308872722688464 valid acc 16/16
Epoch 22 loss 0.2304916512606794 valid acc 16/16
Epoch 22 loss 0.2306316202409792 valid acc 16/16
Epoch 22 loss 0.49105386910586063 valid acc 16/16
Epoch 22 loss 0.3758989348899989 valid acc 16/16
Epoch 22 loss 0.22929827825372828 valid acc 16/16
Epoch 22 loss 0.22695944292637832 valid acc 16/16
Epoch 22 loss 0.27992971130615885 valid acc 15/16
Epoch 22 loss 0.38665697333011206 valid acc 15/16
Epoch 22 loss 0.14144355193055236 valid acc 16/16
Epoch 22 loss 0.1431685715732338 valid acc 16/16
Epoch 22 loss 0.23485012277859207 valid acc 16/16
Epoch 22 loss 0.21432550501130948 valid acc 16/16
Epoch 22 loss 0.24322812166775182 valid acc 16/16
Epoch 22 loss 0.1097243573989331 valid acc 16/16
Epoch 22 loss 0.046317729166142796 valid acc 16/16
Epoch 22 loss 0.46820641795765466 valid acc 16/16
Epoch 22 loss 0.46520803499326946 valid acc 16/16
Epoch 22 loss 0.1720907737120037 valid acc 16/16
Epoch 22 loss 0.08035848266079137 valid acc 16/16
Epoch 22 loss 0.05651482392986984 valid acc 16/16
Epoch 22 loss 0.07442667886996235 valid acc 16/16
Epoch 22 loss 0.08818131535568874 valid acc 16/16
Epoch 22 loss 0.27941124630180597 valid acc 16/16
Epoch 22 loss 0.24377606307050925 valid acc 16/16
Epoch 22 loss 0.1081632609091388 valid acc 16/16
Epoch 22 loss 0.25542447959332626 valid acc 16/16
Epoch 22 loss 0.4389685092489104 valid acc 16/16
Epoch 22 loss 0.0899053292834266 valid acc 16/16
Epoch 22 loss 0.25397204587916833 valid acc 16/16
Epoch 22 loss 0.2483187222277276 valid acc 16/16
Epoch 22 loss 0.11654081405890188 valid acc 16/16
Epoch 22 loss 0.056795797578795215 valid acc 16/16
Epoch 22 loss 0.2661866496699409 valid acc 16/16
Epoch 22 loss 0.41800437704437765 valid acc 16/16
Epoch 22 loss 0.32172099515784797 valid acc 16/16
Epoch 22 loss 0.06149185792081158 valid acc 16/16
Epoch 22 loss 0.07603773809162956 valid acc 16/16
Epoch 22 loss 0.09212140423089143 valid acc 16/16
Epoch 22 loss 0.3091214515459417 valid acc 15/16
Epoch 22 loss 0.13402729025499932 valid acc 16/16
Epoch 22 loss 0.32184005645776737 valid acc 16/16
Epoch 22 loss 0.2571378591968285 valid acc 15/16
Epoch 22 loss 0.10370206254770603 valid acc 16/16
Epoch 22 loss 0.14364600193324306 valid acc 16/16
Epoch 22 loss 0.250679897551272 valid acc 16/16
Epoch 23 loss 0.001302837371395027 valid acc 16/16
Epoch 23 loss 0.2945085441171643 valid acc 16/16
Epoch 23 loss 0.2733307317365241 valid acc 15/16
Epoch 23 loss 0.3399633322677187 valid acc 16/16
Epoch 23 loss 0.20150671587839353 valid acc 15/16
Epoch 23 loss 0.22243379092382531 valid acc 15/16
Epoch 23 loss 0.25916217992195 valid acc 15/16
Epoch 23 loss 0.2211688076280437 valid acc 15/16
Epoch 23 loss 0.11392280466204174 valid acc 15/16
Epoch 23 loss 0.21115479151386574 valid acc 15/16
Epoch 23 loss 0.06362762008810147 valid acc 15/16
Epoch 23 loss 0.2279692067474134 valid acc 15/16
Epoch 23 loss 0.4150181010484427 valid acc 15/16
Epoch 23 loss 0.2058498466434831 valid acc 16/16
Epoch 23 loss 0.22859547751785458 valid acc 16/16
Epoch 23 loss 0.11316411485184641 valid acc 16/16
Epoch 23 loss 0.1740540728140717 valid acc 16/16
Epoch 23 loss 0.1714378410180079 valid acc 15/16
Epoch 23 loss 0.14836601621116702 valid acc 16/16
Epoch 23 loss 0.16070036989128675 valid acc 16/16
Epoch 23 loss 0.5355804315603035 valid acc 15/16
Epoch 23 loss 0.11452824714879473 valid acc 16/16
Epoch 23 loss 0.08071800321375466 valid acc 15/16
Epoch 23 loss 0.22180881717102569 valid acc 16/16
Epoch 23 loss 0.15182576233029482 valid acc 15/16
Epoch 23 loss 0.3712026736419167 valid acc 15/16
Epoch 23 loss 0.07649529166217972 valid acc 15/16
Epoch 23 loss 0.11278016894599097 valid acc 15/16
Epoch 23 loss 0.13717648855149506 valid acc 15/16
Epoch 23 loss 0.09823037248338068 valid acc 15/16
Epoch 23 loss 0.21612340757086057 valid acc 15/16
Epoch 23 loss 0.12621843616190886 valid acc 15/16
Epoch 23 loss 0.13746250993329318 valid acc 15/16
Epoch 23 loss 0.08751600104225027 valid acc 15/16
Epoch 23 loss 0.9152168083353042 valid acc 16/16
Epoch 23 loss 0.38821834753913853 valid acc 16/16
Epoch 23 loss 0.2841590420318094 valid acc 16/16
Epoch 23 loss 0.23885993846023298 valid acc 16/16
Epoch 23 loss 0.28160380247186084 valid acc 16/16
Epoch 23 loss 0.07854806001144177 valid acc 16/16
Epoch 23 loss 0.1751387380649 valid acc 16/16
Epoch 23 loss 0.3159289205787236 valid acc 16/16
Epoch 23 loss 0.04958075963233266 valid acc 16/16
Epoch 23 loss 0.09453976559272353 valid acc 16/16
Epoch 23 loss 0.4945276832811889 valid acc 16/16
Epoch 23 loss 0.07346226730116262 valid acc 16/16
Epoch 23 loss 0.2755228420135837 valid acc 16/16
Epoch 23 loss 0.4726112127718791 valid acc 16/16
Epoch 23 loss 0.33017112529783826 valid acc 16/16
Epoch 23 loss 0.11576929207607273 valid acc 16/16
Epoch 23 loss 0.03246554357931791 valid acc 16/16
Epoch 23 loss 0.3055876752615322 valid acc 16/16
Epoch 23 loss 0.5112889109416856 valid acc 16/16
Epoch 23 loss 0.09410545025645989 valid acc 16/16
Epoch 23 loss 0.18991698068870888 valid acc 16/16
Epoch 23 loss 0.10052728831638269 valid acc 16/16
Epoch 23 loss 0.17972982859862208 valid acc 16/16
Epoch 23 loss 0.10308750547863121 valid acc 16/16
Epoch 23 loss 0.3837930181894594 valid acc 16/16
Epoch 23 loss 0.08757512900835657 valid acc 16/16
Epoch 23 loss 0.09720912546493649 valid acc 16/16
Epoch 23 loss 0.18601457890134637 valid acc 16/16
Epoch 23 loss 0.21446734450884225 valid acc 16/16
Epoch 24 loss 0.002644040238025669 valid acc 16/16
Epoch 24 loss 0.37340924413181664 valid acc 16/16
Epoch 24 loss 0.5152561725739818 valid acc 16/16
Epoch 24 loss 0.36348620222622124 valid acc 16/16
Epoch 24 loss 0.1377724014150969 valid acc 16/16
Epoch 24 loss 0.09136908246653541 valid acc 16/16
Epoch 24 loss 0.21102531956267973 valid acc 15/16
Epoch 24 loss 0.17264012369430093 valid acc 15/16
Epoch 24 loss 0.09120867237168312 valid acc 15/16
Epoch 24 loss 0.1612957150809496 valid acc 15/16
Epoch 24 loss 0.2650998245834593 valid acc 15/16
Epoch 24 loss 0.1883766130979268 valid acc 16/16
Epoch 24 loss 0.31644596977150724 valid acc 16/16
Epoch 24 loss 0.16633579256784498 valid acc 16/16
Epoch 24 loss 0.3895989277236994 valid acc 16/16
Epoch 24 loss 0.0776666560551233 valid acc 16/16
Epoch 24 loss 0.49738739993330516 valid acc 16/16
Epoch 24 loss 0.5064879733006313 valid acc 16/16
Epoch 24 loss 0.1483801061577762 valid acc 16/16
Epoch 24 loss 0.33277872312938356 valid acc 16/16
Epoch 24 loss 0.4093399601257124 valid acc 16/16
Epoch 24 loss 0.03507329161274464 valid acc 16/16
Epoch 24 loss 0.16488388551993666 valid acc 16/16
Epoch 24 loss 0.07846796667829542 valid acc 16/16
Epoch 24 loss 0.326629283919122 valid acc 16/16
Epoch 24 loss 0.30440521152122385 valid acc 16/16
Epoch 24 loss 0.4221156732484605 valid acc 16/16
Epoch 24 loss 0.09314990232190157 valid acc 16/16
Epoch 24 loss 0.2529986293034501 valid acc 15/16
Epoch 24 loss 0.13637813865367804 valid acc 15/16
Epoch 24 loss 0.1273260296556205 valid acc 16/16
Epoch 24 loss 0.17630012207533496 valid acc 16/16
Epoch 24 loss 0.11512323567407406 valid acc 16/16
Epoch 24 loss 0.12281497747289727 valid acc 16/16
Epoch 24 loss 0.7989225428566658 valid acc 16/16
Epoch 24 loss 0.4605609953545154 valid acc 16/16
Epoch 24 loss 0.09105406797520532 valid acc 16/16
Epoch 24 loss 0.18001477197239218 valid acc 16/16
Epoch 24 loss 0.28101518207530973 valid acc 16/16
Epoch 24 loss 0.33329776768917296 valid acc 16/16
Epoch 24 loss 0.2144097875086115 valid acc 16/16
Epoch 24 loss 0.11440058840504158 valid acc 16/16
Epoch 24 loss 0.0470564495213725 valid acc 16/16
Epoch 24 loss 0.1762298152320136 valid acc 16/16
Epoch 24 loss 0.18015136027655443 valid acc 16/16
Epoch 24 loss 0.0698376092806936 valid acc 16/16
Epoch 24 loss 0.19231178542841365 valid acc 16/16
Epoch 24 loss 0.14969015332290309 valid acc 16/16
Epoch 24 loss 0.21525243544004524 valid acc 16/16
Epoch 24 loss 0.10333071986456749 valid acc 15/16
Epoch 24 loss 0.10153026133479559 valid acc 15/16
Epoch 24 loss 0.09447956461802165 valid acc 15/16
Epoch 24 loss 0.21987058960668654 valid acc 16/16
Epoch 24 loss 0.19150143749493326 valid acc 16/16
Epoch 24 loss 0.09195740918475193 valid acc 16/16
Epoch 24 loss 0.13749877570787894 valid acc 16/16
Epoch 24 loss 0.42680386258541836 valid acc 15/16
Epoch 24 loss 0.18259416098044218 valid acc 16/16
Epoch 24 loss 0.3843963965273225 valid acc 16/16
Epoch 24 loss 0.11734809336475432 valid acc 16/16
Epoch 24 loss 0.048969687404798345 valid acc 16/16
Epoch 24 loss 0.2727830296938467 valid acc 16/16
Epoch 24 loss 0.0843454356172918 valid acc 16/16
Epoch 25 loss 0.0034707576825853126 valid acc 16/16
Epoch 25 loss 0.5065366908444499 valid acc 16/16
Epoch 25 loss 0.3042345743694991 valid acc 16/16
Epoch 25 loss 0.1040715952666933 valid acc 16/16
Epoch 25 loss 0.1275424988779716 valid acc 16/16
Epoch 25 loss 0.04304451289335892 valid acc 16/16
Epoch 25 loss 0.2226054610311788 valid acc 16/16
Epoch 25 loss 0.13855114677414715 valid acc 16/16
Epoch 25 loss 0.2838489862397448 valid acc 16/16
Epoch 25 loss 0.13350552886745554 valid acc 16/16
Epoch 25 loss 0.09833583323591621 valid acc 15/16
Epoch 25 loss 0.08233085376707017 valid acc 16/16
Epoch 25 loss 0.23070459693964981 valid acc 16/16
Epoch 25 loss 0.18643906874250235 valid acc 16/16
Epoch 25 loss 0.27108189500315827 valid acc 15/16
Epoch 25 loss 0.18840370393795414 valid acc 15/16
Epoch 25 loss 0.13874325865398285 valid acc 15/16
Epoch 25 loss 0.33666849700133267 valid acc 16/16
Epoch 25 loss 0.07657334696551876 valid acc 16/16
Epoch 25 loss 0.2633628685997307 valid acc 16/16
Epoch 25 loss 0.24824992437774762 valid acc 15/16
Epoch 25 loss 0.0741077967663723 valid acc 15/16
Epoch 25 loss 0.02443672227827509 valid acc 15/16
Epoch 25 loss 0.13320760585914532 valid acc 15/16
Epoch 25 loss 0.34529216101218485 valid acc 15/16
Epoch 25 loss 0.10726918839697178 valid acc 15/16
Epoch 25 loss 0.11056589114632454 valid acc 16/16
Epoch 25 loss 0.030126611117530455 valid acc 16/16
Epoch 25 loss 0.31927845900368107 valid acc 16/16
Epoch 25 loss 0.04538296058734387 valid acc 16/16
Epoch 25 loss 0.0700825165120259 valid acc 16/16
Epoch 25 loss 0.03657689630499045 valid acc 16/16
Epoch 25 loss 0.2236000547436346 valid acc 16/16
Epoch 25 loss 0.4830560601857551 valid acc 15/16
Epoch 25 loss 0.5882745200242651 valid acc 16/16
Epoch 25 loss 0.3057758681249068 valid acc 16/16
Epoch 25 loss 0.1369393708271444 valid acc 16/16
Epoch 25 loss 0.3366255655345766 valid acc 16/16
Epoch 25 loss 0.21905630054347452 valid acc 16/16
Epoch 25 loss 0.06366868667971248 valid acc 15/16
Epoch 25 loss 0.09999142197661914 valid acc 16/16
Epoch 25 loss 0.4573253463562734 valid acc 16/16
Epoch 25 loss 0.13432585521302037 valid acc 16/16
Epoch 25 loss 0.43149331641036603 valid acc 15/16
Epoch 25 loss 0.1585429837424171 valid acc 16/16
Epoch 25 loss 0.08938957984976303 valid acc 16/16
Epoch 25 loss 0.33768344204145745 valid acc 16/16
Epoch 25 loss 0.8886843926296908 valid acc 16/16
Epoch 25 loss 0.06653763245027472 valid acc 16/16
Epoch 25 loss 0.11593066874301355 valid acc 16/16
Epoch 25 loss 0.044146758291362664 valid acc 16/16
Epoch 25 loss 0.1457283023458898 valid acc 16/16
Epoch 25 loss 0.33871143662915965 valid acc 16/16
Epoch 25 loss 0.17572729610395044 valid acc 16/16
Epoch 25 loss 0.4420306671343837 valid acc 15/16
Epoch 25 loss 0.19294851834214194 valid acc 16/16
Epoch 25 loss 0.582549162306427 valid acc 16/16
Epoch 25 loss 0.4421313048722995 valid acc 16/16
Epoch 25 loss 0.4646243432877892 valid acc 16/16
Epoch 25 loss 0.10742057308019154 valid acc 16/16
Epoch 25 loss 0.18384267449395575 valid acc 16/16
Epoch 25 loss 0.21519611566483893 valid acc 16/16
Epoch 25 loss 0.21777153179116818 valid acc 16/16
Epoch 26 loss 0.0057580166535580735 valid acc 16/16
Epoch 26 loss 0.3088856081055273 valid acc 16/16
Epoch 26 loss 0.26118686541472863 valid acc 16/16
Epoch 26 loss 0.09107682719306404 valid acc 16/16
Epoch 26 loss 0.1811383566921113 valid acc 16/16
Epoch 26 loss 0.2115922451967936 valid acc 16/16
Epoch 26 loss 0.1354857277205272 valid acc 16/16
Epoch 26 loss 0.11456614879560117 valid acc 16/16
Epoch 26 loss 0.2426069587986891 valid acc 16/16
Epoch 26 loss 0.14228491504460616 valid acc 16/16
Epoch 26 loss 0.1892155705558004 valid acc 16/16
Epoch 26 loss 0.3726626403259648 valid acc 16/16
Epoch 26 loss 0.48353093279961806 valid acc 16/16
Epoch 26 loss 0.1672999416246463 valid acc 16/16
Epoch 26 loss 0.32411623225678865 valid acc 16/16
Epoch 26 loss 0.1626382935952794 valid acc 16/16
Epoch 26 loss 0.31441623908203814 valid acc 15/16
Epoch 26 loss 0.3255001832940319 valid acc 16/16
Epoch 26 loss 0.2012795943096345 valid acc 16/16
Epoch 26 loss 0.09898008379873964 valid acc 16/16
Epoch 26 loss 0.36598968423179495 valid acc 16/16
Epoch 26 loss 0.1998224336895043 valid acc 16/16
Epoch 26 loss 0.022562310134877583 valid acc 16/16
Epoch 26 loss 0.04683464111728247 valid acc 16/16
Epoch 26 loss 0.13460969997994465 valid acc 16/16
Epoch 26 loss 0.20936468733665514 valid acc 16/16
Epoch 26 loss 0.27921153135092036 valid acc 16/16
Epoch 26 loss 0.26903779313042636 valid acc 16/16
Epoch 26 loss 0.14266555558558675 valid acc 16/16
Epoch 26 loss 0.10795805795520597 valid acc 16/16
Epoch 26 loss 0.36765098822746645 valid acc 16/16
Epoch 26 loss 0.23193409909783425 valid acc 16/16
Epoch 26 loss 0.12218913297313827 valid acc 16/16
Epoch 26 loss 0.16072261716248287 valid acc 16/16
Epoch 26 loss 0.3187321819021469 valid acc 16/16
Epoch 26 loss 0.15885627045254336 valid acc 15/16
Epoch 26 loss 0.1306990715889601 valid acc 16/16
Epoch 26 loss 0.23328088667357222 valid acc 16/16
Epoch 26 loss 0.19451584602247868 valid acc 16/16
Epoch 26 loss 0.0839792625544018 valid acc 16/16
Epoch 26 loss 0.15421761747971313 valid acc 16/16
Epoch 26 loss 0.14246656187988757 valid acc 16/16
Epoch 26 loss 0.1452828175712746 valid acc 16/16
Epoch 26 loss 0.08108796020060455 valid acc 16/16
Epoch 26 loss 0.49048625073531327 valid acc 16/16
Epoch 26 loss 0.0814987430736065 valid acc 16/16
Epoch 26 loss 0.2795043073605593 valid acc 16/16
Epoch 26 loss 0.09186726437869641 valid acc 16/16
Epoch 26 loss 0.13983194360916568 valid acc 16/16
Epoch 26 loss 0.05993998056788996 valid acc 16/16
Epoch 26 loss 0.3014136327472835 valid acc 15/16
Epoch 26 loss 0.1650061157021189 valid acc 16/16
Epoch 26 loss 0.06731691447308419 valid acc 16/16
Epoch 26 loss 0.012088698721775537 valid acc 16/16
Epoch 26 loss 0.12799932861914215 valid acc 16/16
Epoch 26 loss 0.05476343127121411 valid acc 16/16
Epoch 26 loss 0.5983926111105637 valid acc 16/16
Epoch 26 loss 0.09683329375394867 valid acc 16/16
Epoch 26 loss 0.40075174837296906 valid acc 16/16
Epoch 26 loss 0.24485064210838792 valid acc 16/16
Epoch 26 loss 0.07128075516097576 valid acc 16/16
Epoch 26 loss 0.15780434455884232 valid acc 16/16
Epoch 26 loss 0.16754514938219373 valid acc 16/16
Epoch 27 loss 0.0034062560164733746 valid acc 16/16
Epoch 27 loss 0.34449596637066354 valid acc 16/16
Epoch 27 loss 0.43762005603854764 valid acc 15/16
Epoch 27 loss 0.32910835606008404 valid acc 16/16
Epoch 27 loss 0.0556500879138756 valid acc 15/16
Epoch 27 loss 0.16504769725481805 valid acc 16/16
Epoch 27 loss 0.23979790696343 valid acc 16/16
Epoch 27 loss 0.7839166455723824 valid acc 15/16
Epoch 27 loss 0.19705729392577953 valid acc 15/16
Epoch 27 loss 0.0821803230812801 valid acc 15/16
Epoch 27 loss 0.1573600526854308 valid acc 15/16
Epoch 27 loss 0.16115383061108618 valid acc 16/16
Epoch 27 loss 0.21196381213091753 valid acc 16/16
Epoch 27 loss 0.15183261236915324 valid acc 16/16
Epoch 27 loss 0.3711683905069212 valid acc 16/16
Epoch 27 loss 0.10309737789662182 valid acc 16/16
Epoch 27 loss 0.6922416882428177 valid acc 16/16
Epoch 27 loss 0.11678649121670315 valid acc 16/16
Epoch 27 loss 0.18157464663142447 valid acc 16/16
Epoch 27 loss 0.15997388389021966 valid acc 16/16
Epoch 27 loss 0.13977910674858524 valid acc 16/16
Epoch 27 loss 0.1887513668523786 valid acc 16/16
Epoch 27 loss 0.09559197216566259 valid acc 16/16
Epoch 27 loss 0.0677216268422196 valid acc 16/16
Epoch 27 loss 0.09943793316409533 valid acc 16/16
Epoch 27 loss 0.14595535457104025 valid acc 16/16
Epoch 27 loss 0.04387349835842136 valid acc 16/16
Epoch 27 loss 0.029604778395173337 valid acc 16/16
Epoch 27 loss 0.06499106483823575 valid acc 16/16
Epoch 27 loss 0.1345743361574525 valid acc 16/16
Epoch 27 loss 0.04003825842593012 valid acc 16/16
Epoch 27 loss 0.32193842233175807 valid acc 16/16
Epoch 27 loss 0.07239064686872931 valid acc 16/16
Epoch 27 loss 0.07167761162386316 valid acc 16/16
Epoch 27 loss 0.493040559073429 valid acc 16/16
Epoch 27 loss 0.07312800127886254 valid acc 16/16
Epoch 27 loss 0.060532893687109124 valid acc 16/16
Epoch 27 loss 0.3450759253250987 valid acc 16/16
Epoch 27 loss 0.11176218640378094 valid acc 16/16
Epoch 27 loss 0.16696817648391588 valid acc 16/16
Epoch 27 loss 0.27859984381360337 valid acc 16/16
Epoch 27 loss 0.28808377990785367 valid acc 16/16
Epoch 27 loss 0.1748378481774454 valid acc 16/16
Epoch 27 loss 0.09861837366259076 valid acc 16/16
Epoch 27 loss 0.4005096723858793 valid acc 16/16
Epoch 27 loss 0.2500113662177909 valid acc 16/16
Epoch 27 loss 0.19168535846208717 valid acc 16/16
Epoch 27 loss 0.26046250272212895 valid acc 16/16
Epoch 27 loss 0.30167096677382177 valid acc 16/16
Epoch 27 loss 0.061169808264041114 valid acc 16/16
Epoch 27 loss 0.04811873098085462 valid acc 16/16
Epoch 27 loss 0.2681742449201877 valid acc 16/16
Epoch 27 loss 0.23148632999275615 valid acc 15/16
Epoch 27 loss 0.13364761034762662 valid acc 16/16
Epoch 27 loss 0.1823304761935706 valid acc 16/16
Epoch 27 loss 0.14936137890596063 valid acc 16/16
Epoch 27 loss 0.16248050237244982 valid acc 16/16
Epoch 27 loss 0.20600537721836865 valid acc 16/16
Epoch 27 loss 0.2026371576779171 valid acc 16/16
Epoch 27 loss 0.035318535510200644 valid acc 16/16
Epoch 27 loss 0.2754483982904701 valid acc 16/16
Epoch 27 loss 0.19175864339866378 valid acc 16/16
Epoch 27 loss 0.20629241567109347 valid acc 16/16
Epoch 28 loss 0.047805228378934084 valid acc 16/16
Epoch 28 loss 0.465934892998949 valid acc 16/16
Epoch 28 loss 0.5376418938821308 valid acc 15/16
Epoch 28 loss 0.22914553430629508 valid acc 15/16
Epoch 28 loss 0.05701239561189442 valid acc 15/16
Epoch 28 loss 0.08980592085107242 valid acc 15/16
Epoch 28 loss 0.050784246016140455 valid acc 15/16
Epoch 28 loss 0.08570073891918917 valid acc 15/16
Epoch 28 loss 0.2766925505100417 valid acc 15/16
Epoch 28 loss 0.2873631693342335 valid acc 15/16
Epoch 28 loss 0.20980543155021872 valid acc 15/16
Epoch 28 loss 0.3615834956975236 valid acc 16/16
Epoch 28 loss 0.1489584845751637 valid acc 16/16
Epoch 28 loss 0.08414629464105017 valid acc 16/16
Epoch 28 loss 0.08286650189407413 valid acc 16/16
Epoch 28 loss 0.20094687345122092 valid acc 16/16
Epoch 28 loss 0.12001116379287563 valid acc 16/16
Epoch 28 loss 0.37414757912176416 valid acc 15/16
Epoch 28 loss 0.24460144683569346 valid acc 16/16
Epoch 28 loss 0.18482199847246145 valid acc 16/16
Epoch 28 loss 0.3804251272774665 valid acc 15/16
Epoch 28 loss 0.08824831711858716 valid acc 16/16
Epoch 28 loss 0.18401677004703293 valid acc 15/16
Epoch 28 loss 0.2178162112219626 valid acc 16/16
Epoch 28 loss 0.18158782904726417 valid acc 16/16
Epoch 28 loss 0.3435438629914908 valid acc 16/16
Epoch 28 loss 0.27997619949506025 valid acc 16/16
Epoch 28 loss 0.15934096230898492 valid acc 16/16
Epoch 28 loss 0.12191696003309593 valid acc 16/16
Epoch 28 loss 0.06606408333123398 valid acc 16/16
Epoch 28 loss 0.2560415295910866 valid acc 16/16
Epoch 28 loss 0.10282634595814089 valid acc 16/16
Epoch 28 loss 0.15888165745376037 valid acc 16/16
Epoch 28 loss 0.3429918671050744 valid acc 16/16
Epoch 28 loss 0.4294952672043014 valid acc 16/16
Epoch 28 loss 0.13982627534265857 valid acc 16/16
Epoch 28 loss 0.15475578828928738 valid acc 16/16
Epoch 28 loss 0.1280014845413288 valid acc 16/16
Epoch 28 loss 0.12944565368226907 valid acc 16/16
Epoch 28 loss 0.10468201547388722 valid acc 16/16
Epoch 28 loss 0.14645545171960767 valid acc 16/16
Epoch 28 loss 0.09744446891666547 valid acc 16/16
Epoch 28 loss 0.1025142445529906 valid acc 16/16
Epoch 28 loss 0.1275450343455759 valid acc 16/16
Epoch 28 loss 0.1492700458532305 valid acc 16/16
Epoch 28 loss 0.10779397730615814 valid acc 16/16
Epoch 28 loss 0.29951135718789573 valid acc 16/16
Epoch 28 loss 0.2167397832322849 valid acc 16/16
Epoch 28 loss 0.07172860155346367 valid acc 16/16
Epoch 28 loss 0.04905299241204775 valid acc 16/16
Epoch 28 loss 0.1102953362452615 valid acc 16/16
Epoch 28 loss 0.19567912256742587 valid acc 16/16
Epoch 28 loss 0.3083018937945841 valid acc 16/16
Epoch 28 loss 0.06380698169349985 valid acc 16/16
Epoch 28 loss 0.1354120273586864 valid acc 16/16
Epoch 28 loss 0.06855502315531697 valid acc 16/16
Epoch 28 loss 0.2185029224447329 valid acc 16/16
Epoch 28 loss 0.04718893536798635 valid acc 16/16
Epoch 28 loss 0.42397911369771507 valid acc 16/16
Epoch 28 loss 0.1309208871169624 valid acc 16/16
Epoch 28 loss 0.15597928603422861 valid acc 16/16
Epoch 28 loss 0.06671442856707688 valid acc 16/16
Epoch 28 loss 0.19051601889955327 valid acc 16/16
Epoch 29 loss 0.0030141112010617643 valid acc 16/16
Epoch 29 loss 0.20994529832798026 valid acc 16/16
Epoch 29 loss 0.1330319636372112 valid acc 16/16
Epoch 29 loss 0.17603524779067387 valid acc 16/16
Epoch 29 loss 0.06128975904774753 valid acc 16/16
Epoch 29 loss 0.20422488784851764 valid acc 16/16
Epoch 29 loss 0.08823327106043072 valid acc 16/16
Epoch 29 loss 0.3057162766897672 valid acc 15/16
Epoch 29 loss 0.08046039891187431 valid acc 16/16
Epoch 29 loss 0.04786503038490458 valid acc 16/16
Epoch 29 loss 0.06224832050429857 valid acc 16/16
Epoch 29 loss 0.1640804017093112 valid acc 16/16
Epoch 29 loss 0.551007100753202 valid acc 16/16
Epoch 29 loss 0.22149440623390843 valid acc 16/16
Epoch 29 loss 0.3088482422104479 valid acc 16/16
Epoch 29 loss 0.2170313570153983 valid acc 16/16
Epoch 29 loss 0.35031228988659646 valid acc 16/16
Epoch 29 loss 0.1442663900911399 valid acc 16/16
Epoch 29 loss 0.15750889620999625 valid acc 16/16
Epoch 29 loss 0.1612037662005184 valid acc 16/16
Epoch 29 loss 0.5094025961510372 valid acc 16/16
Epoch 29 loss 0.08135783234519922 valid acc 16/16
Epoch 29 loss 0.04308671826824628 valid acc 16/16
Epoch 29 loss 0.03456885865558701 valid acc 16/16
Epoch 29 loss 0.09813725718833832 valid acc 16/16
Epoch 29 loss 0.3697555571177537 valid acc 16/16
Epoch 29 loss 0.15746119104811368 valid acc 16/16
Epoch 29 loss 0.055461291519114664 valid acc 16/16
Epoch 29 loss 0.04901228309868605 valid acc 16/16
Epoch 29 loss 0.02559457318131797 valid acc 16/16
Epoch 29 loss 0.32570890802220726 valid acc 16/16
Epoch 29 loss 0.09110697098431297 valid acc 16/16
Epoch 29 loss 0.0831056909135372 valid acc 16/16
Epoch 29 loss 0.2444346369074518 valid acc 16/16
Epoch 29 loss 0.5057727225736275 valid acc 16/16
Epoch 29 loss 0.4123936441500123 valid acc 16/16
Epoch 29 loss 0.2946593595428565 valid acc 16/16
Epoch 29 loss 0.18810199196136035 valid acc 16/16
Epoch 29 loss 0.15144316333143304 valid acc 16/16
Epoch 29 loss 0.06229483531396851 valid acc 16/16
Epoch 29 loss 0.08698575017040394 valid acc 16/16
Epoch 29 loss 0.10064350842452807 valid acc 16/16
Epoch 29 loss 0.07436776714196092 valid acc 16/16
Epoch 29 loss 0.14505091595873915 valid acc 16/16
Epoch 29 loss 0.09297662571111653 valid acc 16/16
Epoch 29 loss 0.07010165590327833 valid acc 16/16
Epoch 29 loss 0.3065236880711203 valid acc 16/16
Epoch 29 loss 0.13387349746289917 valid acc 16/16
Epoch 29 loss 0.09024792455389358 valid acc 16/16
Epoch 29 loss 0.06516725062051981 valid acc 16/16
Epoch 29 loss 0.11816722693324044 valid acc 16/16
Epoch 29 loss 0.14124260488652318 valid acc 16/16
Epoch 29 loss 0.1815086488341841 valid acc 16/16
Epoch 29 loss 0.033340479863637196 valid acc 16/16
Epoch 29 loss 0.06699455724595599 valid acc 16/16
Epoch 29 loss 0.08045670905566021 valid acc 16/16
Epoch 29 loss 0.2907457872580002 valid acc 16/16
Epoch 29 loss 0.024110195320091454 valid acc 16/16
Epoch 29 loss 0.266580166001503 valid acc 15/16
Epoch 29 loss 0.11946684119570256 valid acc 16/16
Epoch 29 loss 0.3060211025944368 valid acc 15/16
Epoch 29 loss 0.23126953247497467 valid acc 16/16
Epoch 29 loss 0.31890041145951326 valid acc 16/16
Epoch 30 loss 0.00329481291417022 valid acc 16/16
Epoch 30 loss 0.27148811675535867 valid acc 16/16
Epoch 30 loss 0.2568315946386816 valid acc 15/16
Epoch 30 loss 0.42218287190143233 valid acc 15/16
Epoch 30 loss 0.10060086345767005 valid acc 16/16
Epoch 30 loss 0.03408540308160757 valid acc 16/16
Epoch 30 loss 0.2309242063297718 valid acc 15/16
Epoch 30 loss 0.11806656458882397 valid acc 15/16
Epoch 30 loss 0.047170368002421637 valid acc 15/16
Epoch 30 loss 0.08602854472975097 valid acc 16/16
Epoch 30 loss 0.06612169656699207 valid acc 15/16
Epoch 30 loss 0.2896730222421904 valid acc 16/16
Epoch 30 loss 0.25375982898676885 valid acc 16/16
Epoch 30 loss 0.12227946753484642 valid acc 16/16
Epoch 30 loss 0.3013001733402336 valid acc 16/16
Epoch 30 loss 0.04025987317960067 valid acc 16/16
Epoch 30 loss 0.3928854205469492 valid acc 15/16
Epoch 30 loss 0.06501200279035985 valid acc 16/16
Epoch 30 loss 0.29494273476839156 valid acc 16/16
Epoch 30 loss 0.3710778130422687 valid acc 16/16
Epoch 30 loss 0.1940411244081352 valid acc 16/16
Epoch 30 loss 0.3251673630620965 valid acc 16/16
Epoch 30 loss 0.07141592113999967 valid acc 16/16
Epoch 30 loss 0.21283341076619183 valid acc 16/16
Epoch 30 loss 0.16972097566519562 valid acc 16/16
Epoch 30 loss 0.11705896321913284 valid acc 16/16
Epoch 30 loss 0.042734285021581164 valid acc 16/16
Epoch 30 loss 0.17539215045828177 valid acc 16/16
Epoch 30 loss 0.13885357801814052 valid acc 16/16
Epoch 30 loss 0.05282771811747461 valid acc 16/16
Epoch 30 loss 0.10867644396246523 valid acc 15/16
Epoch 30 loss 0.08052449498146075 valid acc 16/16
Epoch 30 loss 0.02005157462738638 valid acc 16/16
Epoch 30 loss 0.09183719523972872 valid acc 16/16
Epoch 30 loss 0.18108509946242857 valid acc 16/16
Epoch 30 loss 0.049362711373303614 valid acc 16/16
Epoch 30 loss 0.03656915038118874 valid acc 16/16
Epoch 30 loss 0.13453283066987926 valid acc 16/16
Epoch 30 loss 0.09698803087062391 valid acc 16/16
Epoch 30 loss 0.22245279394688477 valid acc 16/16
Epoch 30 loss 0.10829030910902593 valid acc 16/16
Epoch 30 loss 0.29758483341095643 valid acc 16/16
Epoch 30 loss 0.25184721360632245 valid acc 16/16
Epoch 30 loss 0.0971389973507342 valid acc 16/16
Epoch 30 loss 0.10010353177649395 valid acc 16/16
Epoch 30 loss 0.27038063108170646 valid acc 16/16
Epoch 30 loss 0.3412507398272412 valid acc 16/16
Epoch 30 loss 0.34182327752189345 valid acc 16/16
Epoch 30 loss 0.40930828156333854 valid acc 16/16
Epoch 30 loss 0.02053092319977501 valid acc 16/16
Epoch 30 loss 0.055604154314166654 valid acc 16/16
Epoch 30 loss 0.07521593906605786 valid acc 16/16
Epoch 30 loss 0.040118194542607566 valid acc 16/16
Epoch 30 loss 0.026259401505456248 valid acc 16/16
Epoch 30 loss 0.10342847948056072 valid acc 16/16
Epoch 30 loss 0.022926040090415045 valid acc 16/16
Epoch 30 loss 0.3820411492932692 valid acc 15/16
Epoch 30 loss 0.087098567814828 valid acc 16/16
Epoch 30 loss 0.11943971443429768 valid acc 16/16
Epoch 30 loss 0.5661475301755466 valid acc 16/16
Epoch 30 loss 0.053927800303959716 valid acc 16/16
Epoch 30 loss 0.1296040877102678 valid acc 16/16
Epoch 30 loss 0.2780375852567652 valid acc 16/16
real    11m56,776s
user    42m23,807s
sys     1m7,513s
```

### avgpool2d + CUDA
```
USE_CUDA_CONV
Epoch 1 loss 2.2984397848187603 valid acc 1/16
Epoch 1 loss 11.511056215178336 valid acc 1/16
Epoch 1 loss 11.531215853835478 valid acc 1/16
Epoch 1 loss 11.509124827322974 valid acc 4/16
Epoch 1 loss 11.447690981453306 valid acc 4/16
Epoch 1 loss 11.352203706960392 valid acc 2/16
Epoch 1 loss 11.271434004211438 valid acc 6/16
Epoch 1 loss 11.124481084729132 valid acc 6/16
Epoch 1 loss 11.046869940758864 valid acc 7/16
Epoch 1 loss 9.630979571834612 valid acc 8/16
Epoch 1 loss 8.862526820149139 valid acc 7/16
Epoch 1 loss 7.7838704738981175 valid acc 9/16
Epoch 1 loss 8.841071928271953 valid acc 14/16
Epoch 1 loss 7.5507883095758315 valid acc 11/16
Epoch 1 loss 7.699341329899247 valid acc 13/16
Epoch 1 loss 5.837680727925946 valid acc 10/16
Epoch 1 loss 8.301657578478276 valid acc 9/16
Epoch 1 loss 6.8532369780478035 valid acc 13/16
Epoch 1 loss 6.570118802367011 valid acc 13/16
Epoch 1 loss 5.299188043517816 valid acc 13/16
Epoch 1 loss 4.862153133349507 valid acc 10/16
Epoch 1 loss 5.291069232909534 valid acc 13/16
Epoch 1 loss 3.779781754143342 valid acc 12/16
Epoch 1 loss 4.723176686061846 valid acc 14/16
Epoch 1 loss 4.388155481591832 valid acc 12/16
Epoch 1 loss 4.369952517907708 valid acc 16/16
Epoch 1 loss 3.9531912371430487 valid acc 12/16
Epoch 1 loss 3.6775845745663878 valid acc 15/16
Epoch 1 loss 2.475860794256662 valid acc 14/16
Epoch 1 loss 2.636275661574205 valid acc 13/16
Epoch 1 loss 4.467045659405666 valid acc 12/16
Epoch 1 loss 3.4755475847955477 valid acc 13/16
Epoch 1 loss 3.862076323559356 valid acc 12/16
Epoch 1 loss 4.714049030868836 valid acc 13/16
Epoch 1 loss 4.578048200008004 valid acc 14/16
Epoch 1 loss 3.7089373703253625 valid acc 13/16
Epoch 1 loss 2.6420701374139814 valid acc 13/16
Epoch 1 loss 3.0841720475030754 valid acc 13/16
Epoch 1 loss 3.4735362293056413 valid acc 15/16
Epoch 1 loss 3.735882788087216 valid acc 14/16
Epoch 1 loss 2.683491479375301 valid acc 15/16
Epoch 1 loss 3.3931221532203186 valid acc 15/16
Epoch 1 loss 2.6519366783069773 valid acc 16/16
Epoch 1 loss 2.8011859260334013 valid acc 15/16
Epoch 1 loss 4.476244705672188 valid acc 16/16
Epoch 1 loss 2.9965227673907613 valid acc 15/16
Epoch 1 loss 3.3464150911125286 valid acc 15/16
Epoch 1 loss 3.714120461726894 valid acc 15/16
Epoch 1 loss 3.480559512298532 valid acc 13/16
Epoch 1 loss 3.1280710388558814 valid acc 16/16
Epoch 1 loss 2.7489085857328197 valid acc 14/16
Epoch 1 loss 3.1653012256749657 valid acc 15/16
Epoch 1 loss 3.2177351112587447 valid acc 15/16
Epoch 1 loss 2.4666192095187176 valid acc 15/16
Epoch 1 loss 3.0441783074986297 valid acc 15/16
Epoch 1 loss 2.5495624019020133 valid acc 15/16
Epoch 1 loss 2.911386056270029 valid acc 12/16
Epoch 1 loss 3.210212299950564 valid acc 14/16
Epoch 1 loss 2.7169406477982436 valid acc 13/16
Epoch 1 loss 2.9149383272863236 valid acc 13/16
Epoch 1 loss 3.089568184604693 valid acc 13/16
Epoch 1 loss 2.9750300820201074 valid acc 12/16
Epoch 1 loss 3.682659324307632 valid acc 13/16
Epoch 2 loss 0.31474424090806413 valid acc 13/16
Epoch 2 loss 3.036200631131896 valid acc 13/16
Epoch 2 loss 2.7859725631539547 valid acc 14/16
Epoch 2 loss 2.841749174518405 valid acc 14/16
Epoch 2 loss 2.7231634810560945 valid acc 13/16
Epoch 2 loss 1.941627967721278 valid acc 14/16
Epoch 2 loss 3.090716260026615 valid acc 14/16
Epoch 2 loss 3.0194131191444145 valid acc 14/16
Epoch 2 loss 3.5136826580902825 valid acc 14/16
Epoch 2 loss 2.778853359095107 valid acc 15/16
Epoch 2 loss 2.096350528340184 valid acc 16/16
Epoch 2 loss 3.2690755953106923 valid acc 16/16
Epoch 2 loss 3.04844544586668 valid acc 15/16
Epoch 2 loss 3.331552242934987 valid acc 14/16
Epoch 2 loss 4.531505596077078 valid acc 15/16
Epoch 2 loss 2.0984843165400924 valid acc 15/16
Epoch 2 loss 3.8133791383776563 valid acc 15/16
Epoch 2 loss 2.9197236640760695 valid acc 13/16
Epoch 2 loss 2.654332857165176 valid acc 15/16
Epoch 2 loss 2.1907780899856815 valid acc 14/16
Epoch 2 loss 2.0627514385417594 valid acc 12/16
Epoch 2 loss 2.858566063062001 valid acc 13/16
Epoch 2 loss 1.5308995003595962 valid acc 14/16
Epoch 2 loss 2.387904584844546 valid acc 15/16
Epoch 2 loss 1.7940790527514099 valid acc 14/16
Epoch 2 loss 1.956481323095013 valid acc 15/16
Epoch 2 loss 2.071085869881103 valid acc 15/16
Epoch 2 loss 1.7474870200301689 valid acc 15/16
Epoch 2 loss 1.8175752801346792 valid acc 14/16
Epoch 2 loss 1.0441702099002814 valid acc 14/16
Epoch 2 loss 2.0089306594324574 valid acc 16/16
Epoch 2 loss 1.5795857093942893 valid acc 14/16
Epoch 2 loss 1.3647799732016583 valid acc 13/16
Epoch 2 loss 1.9476223670681903 valid acc 13/16
Epoch 2 loss 2.6435113407399085 valid acc 15/16
Epoch 2 loss 2.482623757512987 valid acc 13/16
Epoch 2 loss 2.1561743887889397 valid acc 14/16
Epoch 2 loss 2.23822415645346 valid acc 14/16
Epoch 2 loss 2.2137375886041526 valid acc 15/16
Epoch 2 loss 2.6166743567090247 valid acc 15/16
Epoch 2 loss 1.5624595132125354 valid acc 15/16
Epoch 2 loss 2.6810574427667557 valid acc 16/16
Epoch 2 loss 1.7009295308825094 valid acc 16/16
Epoch 2 loss 2.0110663441808487 valid acc 15/16
Epoch 2 loss 2.882685298649342 valid acc 15/16
Epoch 2 loss 1.48572762838683 valid acc 15/16
Epoch 2 loss 2.073766894950605 valid acc 15/16
Epoch 2 loss 2.722809077298554 valid acc 15/16
Epoch 2 loss 2.0843057782301333 valid acc 14/16
Epoch 2 loss 2.0003032282734194 valid acc 14/16
Epoch 2 loss 2.2472776298824813 valid acc 15/16
Epoch 2 loss 2.1492885211622976 valid acc 15/16
Epoch 2 loss 2.129611550111349 valid acc 14/16
Epoch 2 loss 1.6929979171546914 valid acc 14/16
Epoch 2 loss 2.7454058016197127 valid acc 13/16
Epoch 2 loss 1.676601241793715 valid acc 15/16
Epoch 2 loss 1.834944302513394 valid acc 15/16
Epoch 2 loss 2.18596994985457 valid acc 15/16
Epoch 2 loss 2.0152162098471935 valid acc 14/16
Epoch 2 loss 1.208678985313341 valid acc 14/16
Epoch 2 loss 2.0668741321368134 valid acc 13/16
Epoch 2 loss 1.6153864736021808 valid acc 14/16
Epoch 2 loss 2.7372544175965747 valid acc 14/16
Epoch 3 loss 0.16898703881565558 valid acc 14/16
Epoch 3 loss 1.7684492258730438 valid acc 15/16
Epoch 3 loss 1.9396892111964588 valid acc 13/16
Epoch 3 loss 1.8132911977018793 valid acc 14/16
Epoch 3 loss 1.804012469363603 valid acc 15/16
Epoch 3 loss 1.4477899123905753 valid acc 15/16
Epoch 3 loss 1.9418443773730747 valid acc 15/16
Epoch 3 loss 2.109649147687932 valid acc 14/16
Epoch 3 loss 2.3837503446322064 valid acc 15/16
Epoch 3 loss 1.981564784493515 valid acc 15/16
Epoch 3 loss 1.9915452737631647 valid acc 14/16
Epoch 3 loss 2.71310149385725 valid acc 15/16
Epoch 3 loss 2.1369658769762956 valid acc 15/16
Epoch 3 loss 2.53041178937871 valid acc 14/16
Epoch 3 loss 3.2560573154487544 valid acc 14/16
Epoch 3 loss 1.7652467193462358 valid acc 15/16
Epoch 3 loss 3.1736118493862113 valid acc 14/16
Epoch 3 loss 2.804495033660249 valid acc 14/16
Epoch 3 loss 1.7645701516695982 valid acc 15/16
Epoch 3 loss 1.2121575742239727 valid acc 14/16
Epoch 3 loss 1.601860418611769 valid acc 14/16
Epoch 3 loss 1.6124939999264831 valid acc 14/16
Epoch 3 loss 0.663810028177618 valid acc 15/16
Epoch 3 loss 1.919663839661169 valid acc 15/16
Epoch 3 loss 1.5224539891923494 valid acc 14/16
Epoch 3 loss 1.442786804355876 valid acc 14/16
Epoch 3 loss 1.6270127893828459 valid acc 15/16
Epoch 3 loss 1.5454405923704344 valid acc 15/16
Epoch 3 loss 1.6909653830845484 valid acc 15/16
Epoch 3 loss 0.8387991453803862 valid acc 14/16
Epoch 3 loss 1.6299056687288882 valid acc 15/16
Epoch 3 loss 1.2150058872087701 valid acc 14/16
Epoch 3 loss 0.6654422749174542 valid acc 15/16
Epoch 3 loss 0.9727768713795835 valid acc 14/16
Epoch 3 loss 2.564402745946919 valid acc 13/16
Epoch 3 loss 1.9492153503576515 valid acc 14/16
Epoch 3 loss 1.8846246908602704 valid acc 14/16
Epoch 3 loss 1.9145697303228013 valid acc 14/16
Epoch 3 loss 2.049484391812597 valid acc 13/16
Epoch 3 loss 2.1346893192642256 valid acc 15/16
Epoch 3 loss 1.3453242683250268 valid acc 15/16
Epoch 3 loss 1.9408304128461218 valid acc 14/16
Epoch 3 loss 1.4274985814473569 valid acc 15/16
Epoch 3 loss 2.002628550025177 valid acc 14/16
Epoch 3 loss 2.3449886262381696 valid acc 15/16
Epoch 3 loss 1.2366758001704428 valid acc 15/16
Epoch 3 loss 1.5910178435922724 valid acc 15/16
Epoch 3 loss 1.8885837255636144 valid acc 14/16
Epoch 3 loss 1.4720727899786428 valid acc 13/16
Epoch 3 loss 1.701783261675162 valid acc 15/16
Epoch 3 loss 1.971100417501778 valid acc 15/16
Epoch 3 loss 1.6234820077214827 valid acc 15/16
Epoch 3 loss 1.8494501303365272 valid acc 14/16
Epoch 3 loss 1.4352639296420495 valid acc 15/16
Epoch 3 loss 2.150940531983089 valid acc 15/16
Epoch 3 loss 1.2537722059892946 valid acc 15/16
Epoch 3 loss 1.015666947804794 valid acc 15/16
Epoch 3 loss 1.553523768602193 valid acc 15/16
Epoch 3 loss 1.969033652924656 valid acc 15/16
Epoch 3 loss 1.7307370447086876 valid acc 15/16
Epoch 3 loss 1.8082382543419584 valid acc 14/16
Epoch 3 loss 1.5389886783732212 valid acc 14/16
Epoch 3 loss 1.7261219706362134 valid acc 14/16
Epoch 4 loss 0.014905786444186941 valid acc 14/16
Epoch 4 loss 1.7488903921401016 valid acc 15/16
Epoch 4 loss 2.1557312710805983 valid acc 15/16
Epoch 4 loss 1.513586980743501 valid acc 14/16
Epoch 4 loss 1.0814010169860069 valid acc 15/16
Epoch 4 loss 0.8939803905109536 valid acc 15/16
Epoch 4 loss 1.3267740020977792 valid acc 15/16
Epoch 4 loss 1.64477319002981 valid acc 15/16
Epoch 4 loss 1.5549548993943696 valid acc 14/16
Epoch 4 loss 1.0803704844950468 valid acc 14/16
Epoch 4 loss 1.2894141339651521 valid acc 15/16
Epoch 4 loss 1.765968374120795 valid acc 15/16
Epoch 4 loss 1.7354660716070422 valid acc 15/16
Epoch 4 loss 1.8285935714057615 valid acc 15/16
Epoch 4 loss 2.4537391283134102 valid acc 14/16
Epoch 4 loss 1.5648066478137133 valid acc 14/16
Epoch 4 loss 2.6303650510061845 valid acc 15/16
Epoch 4 loss 2.2535603908393904 valid acc 15/16
Epoch 4 loss 1.5798157924812024 valid acc 15/16
Epoch 4 loss 0.7798384192426168 valid acc 15/16
Epoch 4 loss 1.79844491779582 valid acc 14/16
Epoch 4 loss 1.2725605990996378 valid acc 14/16
Epoch 4 loss 0.48242696091814846 valid acc 15/16
Epoch 4 loss 1.5881613836144786 valid acc 15/16
Epoch 4 loss 1.3835835413559128 valid acc 14/16
Epoch 4 loss 0.9444232039645414 valid acc 15/16
Epoch 4 loss 1.0237762635334886 valid acc 15/16
Epoch 4 loss 1.215503329065519 valid acc 15/16
Epoch 4 loss 1.219007236161618 valid acc 15/16
Epoch 4 loss 0.722806771396568 valid acc 15/16
Epoch 4 loss 1.1079572364396566 valid acc 15/16
Epoch 4 loss 0.759795944399567 valid acc 15/16
Epoch 4 loss 0.8441297852579164 valid acc 14/16
Epoch 4 loss 1.5774137288135917 valid acc 14/16
Epoch 4 loss 2.0372466847786193 valid acc 15/16
Epoch 4 loss 1.604243656205606 valid acc 13/16
Epoch 4 loss 1.6596661575312501 valid acc 13/16
Epoch 4 loss 1.1842496149964827 valid acc 15/16
Epoch 4 loss 1.45496640918717 valid acc 15/16
Epoch 4 loss 1.4810302149216872 valid acc 14/16
Epoch 4 loss 1.1202702843701056 valid acc 15/16
Epoch 4 loss 1.0285418581975245 valid acc 15/16
Epoch 4 loss 1.0836432220175525 valid acc 15/16
Epoch 4 loss 0.9477247971227016 valid acc 14/16
Epoch 4 loss 1.8595750306745997 valid acc 16/16
Epoch 4 loss 0.7828671375899114 valid acc 15/16
Epoch 4 loss 1.1103286489297806 valid acc 14/16
Epoch 4 loss 1.6203504532734416 valid acc 14/16
Epoch 4 loss 0.9026282526229084 valid acc 14/16
Epoch 4 loss 0.8053499556829149 valid acc 15/16
Epoch 4 loss 1.0430637922816859 valid acc 15/16
Epoch 4 loss 1.1891398051231679 valid acc 15/16
Epoch 4 loss 1.2073206975943942 valid acc 15/16
Epoch 4 loss 1.0053756997638508 valid acc 15/16
Epoch 4 loss 2.012044449334654 valid acc 15/16
Epoch 4 loss 0.8078842340908327 valid acc 15/16
Epoch 4 loss 1.005454590731855 valid acc 14/16
Epoch 4 loss 0.7472664193197335 valid acc 14/16
Epoch 4 loss 1.6428038054143423 valid acc 14/16
Epoch 4 loss 0.8671184183639001 valid acc 14/16
Epoch 4 loss 1.115902547702423 valid acc 14/16
Epoch 4 loss 1.0313388188080044 valid acc 15/16
Epoch 4 loss 1.1289728909235186 valid acc 15/16
Epoch 5 loss 0.04032970752417997 valid acc 15/16
Epoch 5 loss 1.4705296285461986 valid acc 15/16
Epoch 5 loss 1.676777285828905 valid acc 14/16
Epoch 5 loss 1.1387752146903507 valid acc 14/16
Epoch 5 loss 0.843759599257383 valid acc 14/16
Epoch 5 loss 1.0906290283931184 valid acc 15/16
Epoch 5 loss 1.6285909043230384 valid acc 14/16
Epoch 5 loss 1.42421404611913 valid acc 14/16
Epoch 5 loss 1.3239627558205223 valid acc 15/16
Epoch 5 loss 0.9013112519590488 valid acc 14/16
Epoch 5 loss 1.2156830441326392 valid acc 15/16
Epoch 5 loss 2.3077073681994866 valid acc 14/16
Epoch 5 loss 1.5834002804069263 valid acc 15/16
Epoch 5 loss 1.367400779804428 valid acc 15/16
Epoch 5 loss 2.0828067541818527 valid acc 15/16
Epoch 5 loss 1.91830779937088 valid acc 14/16
Epoch 5 loss 2.300808663618435 valid acc 14/16
Epoch 5 loss 1.8018882764754443 valid acc 15/16
Epoch 5 loss 1.0982430624844746 valid acc 15/16
Epoch 5 loss 0.8872926738835087 valid acc 15/16
Epoch 5 loss 1.6745291166002851 valid acc 14/16
Epoch 5 loss 0.8827955950108124 valid acc 15/16
Epoch 5 loss 0.20050447323135046 valid acc 15/16
Epoch 5 loss 2.244396445880773 valid acc 13/16
Epoch 5 loss 0.8268432596948994 valid acc 14/16
Epoch 5 loss 0.6812144895995205 valid acc 15/16
Epoch 5 loss 1.041835391303116 valid acc 14/16
Epoch 5 loss 0.8203341415103209 valid acc 15/16
Epoch 5 loss 0.8460688071242501 valid acc 15/16
Epoch 5 loss 1.0407850006000419 valid acc 14/16
Epoch 5 loss 1.0129484839124492 valid acc 14/16
Epoch 5 loss 0.5483887512279946 valid acc 14/16
Epoch 5 loss 0.4568427667774658 valid acc 14/16
Epoch 5 loss 0.7653603175258614 valid acc 15/16
Epoch 5 loss 2.1985298608443746 valid acc 16/16
Epoch 5 loss 0.8648136547523007 valid acc 16/16
Epoch 5 loss 1.07594019070679 valid acc 14/16
Epoch 5 loss 0.9205875337426246 valid acc 15/16
Epoch 5 loss 1.3421592436062386 valid acc 16/16
Epoch 5 loss 1.3090589709261842 valid acc 15/16
Epoch 5 loss 1.0067816935200102 valid acc 14/16
Epoch 5 loss 1.047332206313135 valid acc 14/16
Epoch 5 loss 0.8656917351996787 valid acc 16/16
Epoch 5 loss 0.48007035541843457 valid acc 14/16
Epoch 5 loss 1.1897186749958815 valid acc 16/16
Epoch 5 loss 0.5703591897733884 valid acc 16/16
Epoch 5 loss 1.2003943557493177 valid acc 15/16
Epoch 5 loss 1.9491877888297182 valid acc 15/16
Epoch 5 loss 0.6224681440359563 valid acc 15/16
Epoch 5 loss 0.8163345372818627 valid acc 15/16
Epoch 5 loss 0.888599500860899 valid acc 15/16
Epoch 5 loss 0.761245804578845 valid acc 15/16
Epoch 5 loss 1.7521477667967575 valid acc 15/16
Epoch 5 loss 0.7963524379400542 valid acc 15/16
Epoch 5 loss 1.5241740427903032 valid acc 15/16
Epoch 5 loss 0.7910960283240094 valid acc 15/16
Epoch 5 loss 1.2477711513238272 valid acc 14/16
Epoch 5 loss 0.7582186554486046 valid acc 14/16
Epoch 5 loss 1.5479870876534891 valid acc 14/16
Epoch 5 loss 1.0565289206282533 valid acc 14/16
Epoch 5 loss 1.142922615699487 valid acc 14/16
Epoch 5 loss 1.2595418311366369 valid acc 14/16
Epoch 5 loss 1.1504987385957584 valid acc 14/16
Epoch 6 loss 0.08957251333433308 valid acc 14/16
Epoch 6 loss 1.1651249604962648 valid acc 15/16
Epoch 6 loss 0.9062112674194729 valid acc 15/16
Epoch 6 loss 0.8653194903281396 valid acc 15/16
Epoch 6 loss 0.4716688372573421 valid acc 15/16
Epoch 6 loss 0.47356383204793295 valid acc 15/16
Epoch 6 loss 0.985777246261454 valid acc 15/16
Epoch 6 loss 1.3917825473766303 valid acc 15/16
Epoch 6 loss 1.1280814101274177 valid acc 15/16
Epoch 6 loss 1.2711765413570293 valid acc 15/16
Epoch 6 loss 1.3923556479331731 valid acc 15/16
Epoch 6 loss 2.3681745883831224 valid acc 15/16
Epoch 6 loss 2.392958149764458 valid acc 15/16
Epoch 6 loss 1.798788619579518 valid acc 15/16
Epoch 6 loss 1.509205678436411 valid acc 15/16
Epoch 6 loss 1.5824076762977564 valid acc 14/16
Epoch 6 loss 1.9822729769948135 valid acc 15/16
Epoch 6 loss 1.905757100744445 valid acc 15/16
Epoch 6 loss 1.869342438466924 valid acc 15/16
Epoch 6 loss 0.5185824868952765 valid acc 15/16
Epoch 6 loss 1.4599892647618098 valid acc 15/16
Epoch 6 loss 0.5781855974365343 valid acc 15/16
Epoch 6 loss 0.2798982586901687 valid acc 15/16
Epoch 6 loss 1.7587313307936914 valid acc 15/16
Epoch 6 loss 0.689888174701516 valid acc 14/16
Epoch 6 loss 1.1248858431435016 valid acc 15/16
Epoch 6 loss 0.7094374957968861 valid acc 16/16
Epoch 6 loss 0.9073956249461748 valid acc 15/16
Epoch 6 loss 1.1337884294786869 valid acc 15/16
Epoch 6 loss 0.7826871816630776 valid acc 14/16
Epoch 6 loss 1.2730215804945084 valid acc 15/16
Epoch 6 loss 1.1781096425698232 valid acc 15/16
Epoch 6 loss 0.6487529958086533 valid acc 15/16
Epoch 6 loss 0.6105945027655084 valid acc 15/16
Epoch 6 loss 3.1392937418855205 valid acc 15/16
Epoch 6 loss 1.4875956155296752 valid acc 15/16
Epoch 6 loss 1.0854822851769519 valid acc 15/16
Epoch 6 loss 1.0419739084674355 valid acc 15/16
Epoch 6 loss 1.0946349018739725 valid acc 15/16
Epoch 6 loss 1.4247183305694442 valid acc 15/16
Epoch 6 loss 1.2100413323859445 valid acc 15/16
Epoch 6 loss 1.2156282178980602 valid acc 15/16
Epoch 6 loss 0.8248976997633979 valid acc 15/16
Epoch 6 loss 0.6969457331956919 valid acc 15/16
Epoch 6 loss 1.0273676342023932 valid acc 16/16
Epoch 6 loss 0.29026911850120496 valid acc 16/16
Epoch 6 loss 0.8191581829964107 valid acc 14/16
Epoch 6 loss 1.7039926991146825 valid acc 15/16
Epoch 6 loss 0.575793012897492 valid acc 15/16
Epoch 6 loss 0.7752739931165173 valid acc 15/16
Epoch 6 loss 0.5795142972261702 valid acc 15/16
Epoch 6 loss 0.5968846866209156 valid acc 15/16
Epoch 6 loss 1.3428748044948915 valid acc 15/16
Epoch 6 loss 0.5818488685261586 valid acc 15/16
Epoch 6 loss 1.3525757224974435 valid acc 15/16
Epoch 6 loss 0.5370727338958066 valid acc 15/16
Epoch 6 loss 1.117977745560148 valid acc 15/16
Epoch 6 loss 0.927337033193463 valid acc 15/16
Epoch 6 loss 1.1849118545454962 valid acc 14/16
Epoch 6 loss 1.1176952455046303 valid acc 15/16
Epoch 6 loss 0.9432704380754726 valid acc 14/16
Epoch 6 loss 0.8049565642514732 valid acc 14/16
Epoch 6 loss 0.7097781022442455 valid acc 14/16
Epoch 7 loss 0.07842921215709925 valid acc 14/16
Epoch 7 loss 0.967750655158796 valid acc 15/16
Epoch 7 loss 0.9431399555101696 valid acc 15/16
Epoch 7 loss 0.608710252498 valid acc 15/16
Epoch 7 loss 0.48687432324184143 valid acc 15/16
Epoch 7 loss 0.5070021208240316 valid acc 15/16
Epoch 7 loss 1.1596791267548539 valid acc 15/16
Epoch 7 loss 0.9612699057515783 valid acc 15/16
Epoch 7 loss 0.8179486517187052 valid acc 15/16
Epoch 7 loss 0.7133835738206499 valid acc 15/16
Epoch 7 loss 0.6614648637984771 valid acc 15/16
Epoch 7 loss 1.5119081101124978 valid acc 15/16
Epoch 7 loss 1.397476251459444 valid acc 15/16
Epoch 7 loss 1.0238128278591478 valid acc 15/16
Epoch 7 loss 1.106385209928734 valid acc 15/16
Epoch 7 loss 0.9624884226207788 valid acc 15/16
Epoch 7 loss 1.4844867692442474 valid acc 15/16
Epoch 7 loss 1.1857291399348693 valid acc 15/16
Epoch 7 loss 0.9342491965470053 valid acc 15/16
Epoch 7 loss 0.7324579927509849 valid acc 15/16
Epoch 7 loss 1.5113961274104266 valid acc 14/16
Epoch 7 loss 0.6533310539741021 valid acc 15/16
Epoch 7 loss 0.18312001533059882 valid acc 15/16
Epoch 7 loss 0.48563847296580415 valid acc 14/16
Epoch 7 loss 0.8136198373818837 valid acc 14/16
Epoch 7 loss 1.0484752186467226 valid acc 15/16
Epoch 7 loss 0.6498643680545497 valid acc 15/16
Epoch 7 loss 0.8094296483384198 valid acc 15/16
Epoch 7 loss 0.5393044027347049 valid acc 15/16
Epoch 7 loss 0.4411169036737777 valid acc 15/16
Epoch 7 loss 0.9339098383985481 valid acc 14/16
Epoch 7 loss 0.677377316072076 valid acc 14/16
Epoch 7 loss 0.4844283351326989 valid acc 15/16
Epoch 7 loss 0.6790693541780759 valid acc 15/16
Epoch 7 loss 1.6668760928019697 valid acc 15/16
Epoch 7 loss 0.5312119562195059 valid acc 15/16
Epoch 7 loss 0.8885969022676207 valid acc 16/16
Epoch 7 loss 0.5428653441813078 valid acc 16/16
Epoch 7 loss 1.1579763037508344 valid acc 15/16
Epoch 7 loss 0.8311910501139321 valid acc 16/16
Epoch 7 loss 0.2799204550974256 valid acc 16/16
Epoch 7 loss 1.0661289141834884 valid acc 16/16
Epoch 7 loss 0.7927874701992188 valid acc 14/16
Epoch 7 loss 0.5228061859102481 valid acc 15/16
Epoch 7 loss 1.8264116641253858 valid acc 16/16
Epoch 7 loss 0.933594084539191 valid acc 15/16
Epoch 7 loss 0.7229347346405093 valid acc 15/16
Epoch 7 loss 1.5370304472958558 valid acc 15/16
Epoch 7 loss 0.4376015280545155 valid acc 15/16
Epoch 7 loss 0.803497111684526 valid acc 15/16
Epoch 7 loss 0.481714352151928 valid acc 15/16
Epoch 7 loss 0.6772967842616372 valid acc 15/16
Epoch 7 loss 1.404411542727774 valid acc 15/16
Epoch 7 loss 0.9968920771608115 valid acc 15/16
Epoch 7 loss 1.1061727583638532 valid acc 15/16
Epoch 7 loss 0.34758966844801104 valid acc 15/16
Epoch 7 loss 0.5130080419940386 valid acc 15/16
Epoch 7 loss 0.6083510464694134 valid acc 15/16
Epoch 7 loss 1.0041266748040614 valid acc 15/16
Epoch 7 loss 0.6072576109825419 valid acc 14/16
Epoch 7 loss 0.9418511076027666 valid acc 14/16
Epoch 7 loss 0.8063068241317581 valid acc 14/16
Epoch 7 loss 1.1198053950906155 valid acc 15/16
Epoch 8 loss 0.006439518366646746 valid acc 15/16
Epoch 8 loss 0.9965283031894809 valid acc 15/16
Epoch 8 loss 1.1360226263651505 valid acc 15/16
Epoch 8 loss 0.951337114744389 valid acc 15/16
Epoch 8 loss 0.4068906060983687 valid acc 15/16
Epoch 8 loss 0.4951915611819656 valid acc 15/16
Epoch 8 loss 1.007042442981321 valid acc 15/16
Epoch 8 loss 0.9272906621395705 valid acc 15/16
Epoch 8 loss 0.9319824399656316 valid acc 15/16
Epoch 8 loss 0.47854828697292107 valid acc 15/16
Epoch 8 loss 0.6144334733944974 valid acc 15/16
Epoch 8 loss 1.415809646813477 valid acc 15/16
Epoch 8 loss 1.4941972853549084 valid acc 15/16
Epoch 8 loss 1.1023334144664787 valid acc 15/16
Epoch 8 loss 1.2856713880926833 valid acc 15/16
Epoch 8 loss 0.8150157563491494 valid acc 15/16
Epoch 8 loss 1.252383442270299 valid acc 15/16
Epoch 8 loss 1.11077326025834 valid acc 15/16
Epoch 8 loss 0.9466101774012511 valid acc 15/16
Epoch 8 loss 0.7440614824862262 valid acc 15/16
Epoch 8 loss 1.4174101619805146 valid acc 14/16
Epoch 8 loss 0.4536302375841467 valid acc 14/16
Epoch 8 loss 0.3863821943080472 valid acc 14/16
Epoch 8 loss 1.2403530902693407 valid acc 14/16
Epoch 8 loss 0.5826829008043958 valid acc 14/16
Epoch 8 loss 0.7297399057708404 valid acc 15/16
Epoch 8 loss 0.3989502543739868 valid acc 15/16
Epoch 8 loss 0.5049098166764107 valid acc 15/16
Epoch 8 loss 0.6350659591504619 valid acc 15/16
Epoch 8 loss 0.8354917916007858 valid acc 15/16
Epoch 8 loss 0.7318728193743985 valid acc 15/16
Epoch 8 loss 0.44235870709831054 valid acc 14/16
Epoch 8 loss 0.6917907349680537 valid acc 14/16
Epoch 8 loss 0.528437802376344 valid acc 15/16
Epoch 8 loss 1.820706934395944 valid acc 15/16
Epoch 8 loss 0.6052439599448382 valid acc 15/16
Epoch 8 loss 0.34762184086548786 valid acc 15/16
Epoch 8 loss 0.8752381695258525 valid acc 15/16
Epoch 8 loss 1.0504752951514846 valid acc 15/16
Epoch 8 loss 0.5892731473804289 valid acc 15/16
Epoch 8 loss 0.5708974113737753 valid acc 15/16
Epoch 8 loss 0.8950443941368604 valid acc 15/16
Epoch 8 loss 0.4425266767367232 valid acc 15/16
Epoch 8 loss 0.44494146620481495 valid acc 15/16
Epoch 8 loss 1.693814937527692 valid acc 15/16
Epoch 8 loss 0.757335693389174 valid acc 16/16
Epoch 8 loss 0.6903098981569278 valid acc 16/16
Epoch 8 loss 1.226935400901242 valid acc 15/16
Epoch 8 loss 0.7145477527058208 valid acc 15/16
Epoch 8 loss 0.599205947841027 valid acc 15/16
Epoch 8 loss 0.46762195096475434 valid acc 15/16
Epoch 8 loss 0.7148107737975397 valid acc 15/16
Epoch 8 loss 1.027283437211595 valid acc 15/16
Epoch 8 loss 0.7376978626572045 valid acc 15/16
Epoch 8 loss 1.0421903848232454 valid acc 15/16
Epoch 8 loss 0.6214441681386111 valid acc 15/16
Epoch 8 loss 0.9020694181309559 valid acc 15/16
Epoch 8 loss 0.5261115541628695 valid acc 15/16
Epoch 8 loss 0.8480719477051328 valid acc 15/16
Epoch 8 loss 0.747969867421626 valid acc 15/16
Epoch 8 loss 0.6937501766606597 valid acc 14/16
Epoch 8 loss 0.5509587494180019 valid acc 15/16
Epoch 8 loss 0.5861486618794677 valid acc 16/16
Epoch 9 loss 0.01679403770801119 valid acc 16/16
Epoch 9 loss 0.6544181815815124 valid acc 15/16
Epoch 9 loss 0.5351519950554148 valid acc 15/16
Epoch 9 loss 0.5620366973185491 valid acc 15/16
Epoch 9 loss 0.24165878328090995 valid acc 15/16
Epoch 9 loss 0.2876678705984698 valid acc 15/16
Epoch 9 loss 0.6294608231140325 valid acc 15/16
Epoch 9 loss 0.43117312009762526 valid acc 15/16
Epoch 9 loss 0.3948212862449854 valid acc 14/16
Epoch 9 loss 0.24050858871735975 valid acc 15/16
Epoch 9 loss 0.8858298400512872 valid acc 15/16
Epoch 9 loss 1.754570596807633 valid acc 15/16
Epoch 9 loss 0.7588518609994023 valid acc 15/16
Epoch 9 loss 1.1240878301706527 valid acc 15/16
Epoch 9 loss 0.8866329790375727 valid acc 15/16
Epoch 9 loss 0.5961292363071595 valid acc 14/16
Epoch 9 loss 1.1794526948616488 valid acc 15/16
Epoch 9 loss 1.2554067807318106 valid acc 15/16
Epoch 9 loss 1.2017826684675048 valid acc 15/16
Epoch 9 loss 0.5733491510216977 valid acc 15/16
Epoch 9 loss 1.2562445505402058 valid acc 15/16
Epoch 9 loss 0.33082604634995144 valid acc 15/16
Epoch 9 loss 0.19677961449753106 valid acc 15/16
Epoch 9 loss 0.5924947410461432 valid acc 14/16
Epoch 9 loss 0.3321738120682368 valid acc 14/16
Epoch 9 loss 0.6859698003548316 valid acc 15/16
Epoch 9 loss 0.47076243100330734 valid acc 15/16
Epoch 9 loss 0.564129587284874 valid acc 15/16
Epoch 9 loss 0.6876317527617429 valid acc 15/16
Epoch 9 loss 0.61167223864685 valid acc 14/16
Epoch 9 loss 0.807618348303591 valid acc 14/16
Epoch 9 loss 0.49623399222653136 valid acc 14/16
Epoch 9 loss 0.36811823313463754 valid acc 15/16
Epoch 9 loss 0.4390413011273304 valid acc 15/16
Epoch 9 loss 1.4524046463314837 valid acc 15/16
Epoch 9 loss 0.25303819712864367 valid acc 16/16
Epoch 9 loss 0.3321623658130027 valid acc 16/16
Epoch 9 loss 1.2140482672562778 valid acc 16/16
Epoch 9 loss 1.0239236852833717 valid acc 15/16
Epoch 9 loss 0.5956781014298346 valid acc 15/16
Epoch 9 loss 0.3319408263285862 valid acc 15/16
Epoch 9 loss 0.7423475907415611 valid acc 15/16
Epoch 9 loss 0.5337928213312172 valid acc 15/16
Epoch 9 loss 0.2681098300613677 valid acc 15/16
Epoch 9 loss 0.6051490100262775 valid acc 15/16
Epoch 9 loss 0.2828583865603356 valid acc 15/16
Epoch 9 loss 0.7060956416806945 valid acc 15/16
Epoch 9 loss 1.4016778768931137 valid acc 15/16
Epoch 9 loss 0.9175609576354558 valid acc 15/16
Epoch 9 loss 0.49002953364856344 valid acc 15/16
Epoch 9 loss 0.44189000769668974 valid acc 15/16
Epoch 9 loss 0.48597388964991156 valid acc 15/16
Epoch 9 loss 1.0798381178162166 valid acc 15/16
Epoch 9 loss 0.33915220479945696 valid acc 15/16
Epoch 9 loss 0.9769456195002207 valid acc 15/16
Epoch 9 loss 0.7708729160611245 valid acc 15/16
Epoch 9 loss 0.7114960190545243 valid acc 15/16
Epoch 9 loss 0.34359363536060694 valid acc 15/16
Epoch 9 loss 0.9859462441913387 valid acc 15/16
Epoch 9 loss 0.39089649293230816 valid acc 15/16
Epoch 9 loss 0.5492931475908142 valid acc 14/16
Epoch 9 loss 0.6943954360095385 valid acc 15/16
Epoch 9 loss 0.39949279866145704 valid acc 15/16
Epoch 10 loss 0.027368677415479603 valid acc 15/16
Epoch 10 loss 0.7158171587928396 valid acc 15/16
Epoch 10 loss 0.8591818322943401 valid acc 15/16
Epoch 10 loss 0.46848549041315646 valid acc 15/16
Epoch 10 loss 0.1756087971501748 valid acc 15/16
Epoch 10 loss 0.4629758525569142 valid acc 15/16
Epoch 10 loss 0.6071356142269303 valid acc 15/16
Epoch 10 loss 0.9807317630969496 valid acc 15/16
Epoch 10 loss 0.3371913044600354 valid acc 15/16
Epoch 10 loss 0.4691500740426059 valid acc 15/16
Epoch 10 loss 0.6058626294087974 valid acc 15/16
Epoch 10 loss 2.047071904803176 valid acc 15/16
Epoch 10 loss 0.7987661332814202 valid acc 15/16
Epoch 10 loss 1.2415555274054957 valid acc 15/16
Epoch 10 loss 1.186622165442019 valid acc 15/16
Epoch 10 loss 0.4292059296588838 valid acc 15/16
Epoch 10 loss 1.0055517623493553 valid acc 14/16
Epoch 10 loss 1.1971164670743324 valid acc 15/16
Epoch 10 loss 0.7809709782046717 valid acc 15/16
Epoch 10 loss 0.2518801439010874 valid acc 15/16
Epoch 10 loss 1.0637495653461282 valid acc 14/16
Epoch 10 loss 0.3119717823306708 valid acc 15/16
Epoch 10 loss 0.2344268651242366 valid acc 15/16
Epoch 10 loss 0.8447930922664473 valid acc 15/16
Epoch 10 loss 0.45090289290182695 valid acc 15/16
Epoch 10 loss 0.53985214987863 valid acc 15/16
Epoch 10 loss 0.3302257735644505 valid acc 15/16
Epoch 10 loss 0.700010199761705 valid acc 15/16
Epoch 10 loss 0.838768445061387 valid acc 15/16
Epoch 10 loss 0.30702697056223194 valid acc 15/16
Epoch 10 loss 0.5271599246554814 valid acc 14/16
Epoch 10 loss 0.6886258514288571 valid acc 15/16
Epoch 10 loss 0.24648960233029402 valid acc 15/16
Epoch 10 loss 0.3416646109629185 valid acc 15/16
Epoch 10 loss 1.58272466798751 valid acc 15/16
Epoch 10 loss 0.5363689105885017 valid acc 15/16
Epoch 10 loss 0.5650059079608378 valid acc 15/16
Epoch 10 loss 0.7601329222066529 valid acc 16/16
Epoch 10 loss 0.6835435760143171 valid acc 15/16
Epoch 10 loss 0.44850878851665166 valid acc 16/16
Epoch 10 loss 0.1324104228215548 valid acc 15/16
Epoch 10 loss 0.31941362619097363 valid acc 15/16
Epoch 10 loss 0.5602169924160227 valid acc 15/16
Epoch 10 loss 0.16551809301276138 valid acc 15/16
Epoch 10 loss 0.7491117630652446 valid acc 15/16
Epoch 10 loss 0.23978626159103958 valid acc 15/16
Epoch 10 loss 0.5404323939146431 valid acc 15/16
Epoch 10 loss 0.806628931082233 valid acc 15/16
Epoch 10 loss 0.8326735875293689 valid acc 15/16
Epoch 10 loss 0.4821805097534082 valid acc 15/16
Epoch 10 loss 0.2096945746804445 valid acc 15/16
Epoch 10 loss 0.3594215955489515 valid acc 15/16
Epoch 10 loss 0.6453040406593903 valid acc 15/16
Epoch 10 loss 0.15165902782138768 valid acc 15/16
Epoch 10 loss 0.9975630233168982 valid acc 15/16
Epoch 10 loss 0.4825479228036793 valid acc 15/16
Epoch 10 loss 0.6619482229818817 valid acc 14/16
Epoch 10 loss 0.36138452459533277 valid acc 15/16
Epoch 10 loss 0.7668784587899014 valid acc 15/16
Epoch 10 loss 0.7632024661736858 valid acc 15/16
Epoch 10 loss 0.3419440394589019 valid acc 14/16
Epoch 10 loss 0.5114906241881001 valid acc 15/16
Epoch 10 loss 0.4501661443597031 valid acc 15/16
Epoch 11 loss 0.015459129051685394 valid acc 14/16
Epoch 11 loss 0.5188834191659546 valid acc 14/16
Epoch 11 loss 0.46154970555868663 valid acc 15/16
Epoch 11 loss 0.5040283973322939 valid acc 15/16
Epoch 11 loss 0.24204862625378054 valid acc 15/16
Epoch 11 loss 0.39794218524385916 valid acc 15/16
Epoch 11 loss 0.8129014200796827 valid acc 15/16
Epoch 11 loss 0.8075347491210372 valid acc 15/16
Epoch 11 loss 0.4271534261052353 valid acc 15/16
Epoch 11 loss 0.34443325652324674 valid acc 15/16
Epoch 11 loss 0.46259841789355416 valid acc 15/16
Epoch 11 loss 1.3607281562582045 valid acc 15/16
Epoch 11 loss 0.9705352510600215 valid acc 15/16
Epoch 11 loss 0.5342816320398424 valid acc 15/16
Epoch 11 loss 1.152293046473444 valid acc 15/16
Epoch 11 loss 0.5965409359478226 valid acc 15/16
Epoch 11 loss 0.7959286175079223 valid acc 15/16
Epoch 11 loss 0.5393771541605934 valid acc 15/16
Epoch 11 loss 0.9246365539713013 valid acc 15/16
Epoch 11 loss 0.8449954856496822 valid acc 15/16
Epoch 11 loss 0.8250807242660362 valid acc 14/16
Epoch 11 loss 0.44691097625330706 valid acc 15/16
Epoch 11 loss 0.20908510645868264 valid acc 15/16
Epoch 11 loss 0.5446281228532028 valid acc 15/16
Epoch 11 loss 0.4167367948392395 valid acc 15/16
Epoch 11 loss 0.6170327901161572 valid acc 15/16
Epoch 11 loss 0.3127930688710066 valid acc 15/16
Epoch 11 loss 0.5265971545487991 valid acc 15/16
Epoch 11 loss 0.3122686417635685 valid acc 15/16
Epoch 11 loss 0.11006876834534646 valid acc 15/16
Epoch 11 loss 0.5247246851535587 valid acc 15/16
Epoch 11 loss 0.5184089028432288 valid acc 14/16
Epoch 11 loss 0.27509927102212506 valid acc 15/16
Epoch 11 loss 0.363212622770925 valid acc 15/16
Epoch 11 loss 1.4594969408259604 valid acc 16/16
Epoch 11 loss 0.870179726631012 valid acc 15/16
Epoch 11 loss 0.22258990728687125 valid acc 15/16
Epoch 11 loss 0.41770004661387317 valid acc 15/16
Epoch 11 loss 0.9273387791113004 valid acc 15/16
Epoch 11 loss 0.6052907192260769 valid acc 15/16
Epoch 11 loss 0.3744479592771305 valid acc 15/16
Epoch 11 loss 0.48006664839059643 valid acc 16/16
Epoch 11 loss 0.9823890701360106 valid acc 15/16
Epoch 11 loss 0.2854149961433126 valid acc 15/16
Epoch 11 loss 0.7207551700973388 valid acc 16/16
Epoch 11 loss 0.7784497820332997 valid acc 16/16
Epoch 11 loss 0.6814421195169355 valid acc 15/16
Epoch 11 loss 1.2440522736623878 valid acc 15/16
Epoch 11 loss 0.3870981987228445 valid acc 15/16
Epoch 11 loss 0.43907477435848724 valid acc 15/16
Epoch 11 loss 0.6010086379277348 valid acc 15/16
Epoch 11 loss 0.6552634745235775 valid acc 15/16
Epoch 11 loss 0.746469130770276 valid acc 15/16
Epoch 11 loss 0.5390739563462988 valid acc 15/16
Epoch 11 loss 0.6769884581140849 valid acc 15/16
Epoch 11 loss 0.4465233893294751 valid acc 15/16
Epoch 11 loss 0.5829900661361689 valid acc 15/16
Epoch 11 loss 0.3735690753941526 valid acc 14/16
Epoch 11 loss 0.7125601313456824 valid acc 14/16
Epoch 11 loss 0.4851067387859307 valid acc 15/16
Epoch 11 loss 0.5317487267610959 valid acc 14/16
Epoch 11 loss 0.37093138802997383 valid acc 14/16
Epoch 11 loss 0.32379795052045784 valid acc 14/16
Epoch 12 loss 0.17640379487927638 valid acc 14/16
Epoch 12 loss 0.5010294680229805 valid acc 14/16
Epoch 12 loss 0.6750980707929666 valid acc 14/16
Epoch 12 loss 0.6018520303680217 valid acc 15/16
Epoch 12 loss 0.15063135865314436 valid acc 15/16
Epoch 12 loss 0.43696697869878554 valid acc 15/16
Epoch 12 loss 0.5985869228603538 valid acc 15/16
Epoch 12 loss 0.7554593670165535 valid acc 14/16
Epoch 12 loss 0.6633177710659455 valid acc 14/16
Epoch 12 loss 0.43391880444770453 valid acc 14/16
Epoch 12 loss 0.33217036684293466 valid acc 14/16
Epoch 12 loss 0.9835853074270678 valid acc 14/16
Epoch 12 loss 0.8909027533017937 valid acc 14/16
Epoch 12 loss 0.5587282357181561 valid acc 14/16
Epoch 12 loss 0.9419513073066472 valid acc 14/16
Epoch 12 loss 0.8963306625955811 valid acc 15/16
Epoch 12 loss 0.5752606843452175 valid acc 15/16
Epoch 12 loss 0.8577846078292285 valid acc 15/16
Epoch 12 loss 0.9200817531481573 valid acc 14/16
Epoch 12 loss 0.28778297469824854 valid acc 15/16
Epoch 12 loss 1.100230545424327 valid acc 15/16
Epoch 12 loss 0.38882491203088415 valid acc 14/16
Epoch 12 loss 0.12978353646819918 valid acc 14/16
Epoch 12 loss 0.6607209273712462 valid acc 14/16
Epoch 12 loss 0.39564592540652244 valid acc 15/16
Epoch 12 loss 0.4361950422285864 valid acc 15/16
Epoch 12 loss 0.21687379908781343 valid acc 15/16
Epoch 12 loss 0.3382955902419882 valid acc 15/16
Epoch 12 loss 0.47343407575802543 valid acc 15/16
Epoch 12 loss 0.1010213431550876 valid acc 15/16
Epoch 12 loss 0.535838527941743 valid acc 15/16
Epoch 12 loss 0.6994767844439317 valid acc 15/16
Epoch 12 loss 0.37610211222599216 valid acc 15/16
Epoch 12 loss 0.27311761777564664 valid acc 15/16
Epoch 12 loss 1.3910236749099334 valid acc 15/16
Epoch 12 loss 0.5532813388394274 valid acc 15/16
Epoch 12 loss 0.15297925339054075 valid acc 16/16
Epoch 12 loss 0.15806837155223258 valid acc 15/16
Epoch 12 loss 0.7967930999586674 valid acc 15/16
Epoch 12 loss 0.37525327305976863 valid acc 15/16
Epoch 12 loss 0.49877924132208806 valid acc 15/16
Epoch 12 loss 1.1189264516569786 valid acc 15/16
Epoch 12 loss 0.4628068160344693 valid acc 15/16
Epoch 12 loss 0.3128992806809117 valid acc 15/16
Epoch 12 loss 0.2679826983147353 valid acc 15/16
Epoch 12 loss 0.6827686437669218 valid acc 15/16
Epoch 12 loss 0.548125993264613 valid acc 15/16
Epoch 12 loss 0.7652737826354343 valid acc 15/16
Epoch 12 loss 0.4845430521956652 valid acc 15/16
Epoch 12 loss 0.15852529474326 valid acc 15/16
Epoch 12 loss 0.44274454988954304 valid acc 15/16
Epoch 12 loss 0.35856554076571445 valid acc 15/16
Epoch 12 loss 0.2635528119232702 valid acc 15/16
Epoch 12 loss 0.1512746929151071 valid acc 15/16
Epoch 12 loss 0.4462981822262254 valid acc 15/16
Epoch 12 loss 0.3987634206283463 valid acc 15/16
Epoch 12 loss 0.4599349143364508 valid acc 14/16
Epoch 12 loss 0.36553412897917453 valid acc 14/16
Epoch 12 loss 0.672943569318671 valid acc 15/16
Epoch 12 loss 0.2742522421031633 valid acc 14/16
Epoch 12 loss 0.5564044254623022 valid acc 14/16
Epoch 12 loss 0.3056887870441428 valid acc 14/16
Epoch 12 loss 0.3811697190824969 valid acc 14/16
Epoch 13 loss 0.027857211406040694 valid acc 14/16
Epoch 13 loss 0.608853195269562 valid acc 15/16
Epoch 13 loss 0.6167224175568102 valid acc 15/16
Epoch 13 loss 0.5655134891442668 valid acc 15/16
Epoch 13 loss 0.3195839524473607 valid acc 15/16
Epoch 13 loss 0.5466418798638238 valid acc 15/16
Epoch 13 loss 0.5594055131247315 valid acc 15/16
Epoch 13 loss 0.6069262237076098 valid acc 15/16
Epoch 13 loss 0.42132588508625235 valid acc 15/16
Epoch 13 loss 0.5084121354104094 valid acc 14/16
Epoch 13 loss 0.6367662681766533 valid acc 15/16
Epoch 13 loss 1.1884325738718142 valid acc 15/16
Epoch 13 loss 1.0064121092152032 valid acc 15/16
Epoch 13 loss 0.7579541392220697 valid acc 15/16
Epoch 13 loss 0.7254186073006017 valid acc 15/16
Epoch 13 loss 0.6760159909494434 valid acc 15/16
Epoch 13 loss 0.6839655755547503 valid acc 14/16
Epoch 13 loss 0.5455789255234693 valid acc 15/16
Epoch 13 loss 0.4442901991444812 valid acc 15/16
Epoch 13 loss 0.5089195907027113 valid acc 15/16
Epoch 13 loss 1.2329349789489985 valid acc 14/16
Epoch 13 loss 0.3861287849210716 valid acc 15/16
Epoch 13 loss 0.09179570448594704 valid acc 14/16
Epoch 13 loss 1.0781695422511883 valid acc 14/16
Epoch 13 loss 0.4237851237763342 valid acc 15/16
Epoch 13 loss 0.39534359483476045 valid acc 15/16
Epoch 13 loss 0.3336683211716999 valid acc 14/16
Epoch 13 loss 0.3610444511168733 valid acc 15/16
Epoch 13 loss 0.42072145286424073 valid acc 15/16
Epoch 13 loss 0.3201443991004734 valid acc 14/16
Epoch 13 loss 0.24336810461357722 valid acc 14/16
Epoch 13 loss 0.38103755548928264 valid acc 14/16
Epoch 13 loss 0.1361036639816789 valid acc 14/16
Epoch 13 loss 0.22480093582636948 valid acc 14/16
Epoch 13 loss 1.038708960510028 valid acc 14/16
Epoch 13 loss 0.3743277910604082 valid acc 15/16
Epoch 13 loss 0.3535366789642616 valid acc 15/16
Epoch 13 loss 0.3494736101492709 valid acc 16/16
Epoch 13 loss 0.3153603726315476 valid acc 15/16
Epoch 13 loss 0.3747726708520255 valid acc 15/16
Epoch 13 loss 0.4184786813564419 valid acc 15/16
Epoch 13 loss 0.5770786419309102 valid acc 15/16
Epoch 13 loss 0.4094670991858038 valid acc 15/16
Epoch 13 loss 0.13664530968070704 valid acc 15/16
Epoch 13 loss 0.4555728064417205 valid acc 15/16
Epoch 13 loss 0.3415458592992553 valid acc 15/16
Epoch 13 loss 0.36624401762770775 valid acc 15/16
Epoch 13 loss 0.5502471382826483 valid acc 15/16
Epoch 13 loss 0.23002831814472074 valid acc 15/16
Epoch 13 loss 0.356008390812947 valid acc 15/16
Epoch 13 loss 0.1332331396740326 valid acc 15/16
Epoch 13 loss 0.2840368773679466 valid acc 15/16
Epoch 13 loss 0.34783929918261336 valid acc 16/16
Epoch 13 loss 0.6319419890860258 valid acc 15/16
Epoch 13 loss 0.6628011285998114 valid acc 15/16
Epoch 13 loss 0.3758464089715691 valid acc 15/16
Epoch 13 loss 0.889530159053312 valid acc 15/16
Epoch 13 loss 0.8035701438979694 valid acc 14/16
Epoch 13 loss 0.9464896383560986 valid acc 14/16
Epoch 13 loss 0.5008138853489208 valid acc 15/16
Epoch 13 loss 0.3886033713766559 valid acc 14/16
Epoch 13 loss 0.2352599355811375 valid acc 14/16
Epoch 13 loss 0.20939999646401808 valid acc 15/16
Epoch 14 loss 0.00905944245442214 valid acc 15/16
Epoch 14 loss 0.4301421094108035 valid acc 15/16
Epoch 14 loss 0.6273033086272575 valid acc 15/16
Epoch 14 loss 0.35071821596654973 valid acc 15/16
Epoch 14 loss 0.45932182454422116 valid acc 15/16
Epoch 14 loss 0.32030424236301547 valid acc 15/16
Epoch 14 loss 0.5244436111533151 valid acc 15/16
Epoch 14 loss 0.7225897095848628 valid acc 15/16
Epoch 14 loss 0.5001509848021357 valid acc 15/16
Epoch 14 loss 0.36510340345935816 valid acc 15/16
Epoch 14 loss 0.2256129069295015 valid acc 15/16
Epoch 14 loss 0.39778302370576946 valid acc 15/16
Epoch 14 loss 0.31580108555257286 valid acc 15/16
Epoch 14 loss 0.5160925236875108 valid acc 15/16
Epoch 14 loss 0.5728022047879884 valid acc 15/16
Epoch 14 loss 0.45818153223308067 valid acc 15/16
Epoch 14 loss 0.753534981935684 valid acc 15/16
Epoch 14 loss 0.5935659598582033 valid acc 15/16
Epoch 14 loss 0.6373380455355484 valid acc 15/16
Epoch 14 loss 0.35696554473615477 valid acc 15/16
Epoch 14 loss 0.7802518199245256 valid acc 14/16
Epoch 14 loss 0.37509942854653383 valid acc 14/16
Epoch 14 loss 0.10094987369496278 valid acc 14/16
Epoch 14 loss 0.44589159423724184 valid acc 14/16
Epoch 14 loss 0.29411052974259944 valid acc 15/16
Epoch 14 loss 0.35439791988497327 valid acc 15/16
Epoch 14 loss 0.21676275973739104 valid acc 15/16
Epoch 14 loss 0.22646131221644394 valid acc 15/16
Epoch 14 loss 0.21544802613800645 valid acc 15/16
Epoch 14 loss 0.32662412245616385 valid acc 15/16
Epoch 14 loss 0.2999111255705572 valid acc 14/16
Epoch 14 loss 0.2911443363610683 valid acc 14/16
Epoch 14 loss 0.2750946358048167 valid acc 14/16
Epoch 14 loss 0.5708531394653465 valid acc 15/16
Epoch 14 loss 0.7866256720085587 valid acc 15/16
Epoch 14 loss 0.7954136344106941 valid acc 15/16
Epoch 14 loss 0.7273688435729041 valid acc 16/16
Epoch 14 loss 0.4669581639728691 valid acc 16/16
Epoch 14 loss 0.6667072286337696 valid acc 16/16
Epoch 14 loss 0.48974534713828216 valid acc 15/16
Epoch 14 loss 0.1633410861736771 valid acc 15/16
Epoch 14 loss 0.5536309833115611 valid acc 16/16
Epoch 14 loss 0.36141000708800897 valid acc 15/16
Epoch 14 loss 0.5526521968250788 valid acc 15/16
Epoch 14 loss 0.4139700379359851 valid acc 15/16
Epoch 14 loss 0.17219988976980768 valid acc 16/16
Epoch 14 loss 0.672374424754913 valid acc 15/16
Epoch 14 loss 0.9201891115125249 valid acc 15/16
Epoch 14 loss 0.4319490410515138 valid acc 15/16
Epoch 14 loss 0.07648615662305247 valid acc 15/16
Epoch 14 loss 0.49011679856329315 valid acc 15/16
Epoch 14 loss 0.2696893661930749 valid acc 15/16
Epoch 14 loss 0.6854342529284689 valid acc 15/16
Epoch 14 loss 0.21633025165176972 valid acc 15/16
Epoch 14 loss 0.6268551223584433 valid acc 15/16
Epoch 14 loss 0.4079897423741624 valid acc 15/16
Epoch 14 loss 0.5733497509729871 valid acc 15/16
Epoch 14 loss 0.421776075695971 valid acc 15/16
Epoch 14 loss 0.5538183514559374 valid acc 15/16
Epoch 14 loss 0.5438062152406229 valid acc 15/16
Epoch 14 loss 0.4041233250768729 valid acc 15/16
Epoch 14 loss 0.305754730797976 valid acc 14/16
Epoch 14 loss 0.46158700430939453 valid acc 14/16
Epoch 15 loss 0.0048947416250921805 valid acc 15/16
Epoch 15 loss 0.32279078426872876 valid acc 15/16
Epoch 15 loss 0.5232119105970385 valid acc 15/16
Epoch 15 loss 0.4633502882187095 valid acc 15/16
Epoch 15 loss 0.16601561172098034 valid acc 15/16
Epoch 15 loss 0.3047730319681647 valid acc 15/16
Epoch 15 loss 0.46741490388860574 valid acc 15/16
Epoch 15 loss 0.6122686293029056 valid acc 15/16
Epoch 15 loss 0.25487781512330226 valid acc 15/16
Epoch 15 loss 0.2788493705301944 valid acc 15/16
Epoch 15 loss 0.43605394586941687 valid acc 15/16
Epoch 15 loss 0.9409317233531954 valid acc 15/16
Epoch 15 loss 0.7973522171850358 valid acc 15/16
Epoch 15 loss 0.4091273462851381 valid acc 15/16
Epoch 15 loss 0.6300434719533936 valid acc 15/16
Epoch 15 loss 0.5510480209223054 valid acc 15/16
Epoch 15 loss 0.8051687174722545 valid acc 15/16
Epoch 15 loss 0.6086370956865159 valid acc 15/16
Epoch 15 loss 0.36956433212593637 valid acc 15/16
Epoch 15 loss 0.3928154249702993 valid acc 15/16
Epoch 15 loss 0.7346414591406579 valid acc 14/16
Epoch 15 loss 0.5397514968329412 valid acc 14/16
Epoch 15 loss 0.2344480277163989 valid acc 14/16
Epoch 15 loss 0.3590822381525845 valid acc 14/16
Epoch 15 loss 0.6176481051773923 valid acc 15/16
Epoch 15 loss 1.107407357217634 valid acc 15/16
Epoch 15 loss 0.5198484061464481 valid acc 15/16
Epoch 15 loss 0.2837793337642914 valid acc 15/16
Epoch 15 loss 0.34528948249361663 valid acc 15/16
Epoch 15 loss 0.2608595688467499 valid acc 15/16
Epoch 15 loss 0.43690643595376855 valid acc 14/16
Epoch 15 loss 0.39002856511911294 valid acc 14/16
Epoch 15 loss 0.24030671512040547 valid acc 15/16
Epoch 15 loss 0.30944126305551134 valid acc 15/16
Epoch 15 loss 0.8355245104709736 valid acc 15/16
Epoch 15 loss 0.2849500227874493 valid acc 15/16
Epoch 15 loss 0.40564506730152394 valid acc 16/16
Epoch 15 loss 0.24794027395567114 valid acc 15/16
Epoch 15 loss 0.44583445946057887 valid acc 15/16
Epoch 15 loss 0.18189666128739365 valid acc 15/16
Epoch 15 loss 0.15226416379237118 valid acc 15/16
Epoch 15 loss 0.5668033652748702 valid acc 15/16
Epoch 15 loss 0.5788232442363146 valid acc 15/16
Epoch 15 loss 0.2187860450623553 valid acc 15/16
Epoch 15 loss 0.32280780104483175 valid acc 15/16
Epoch 15 loss 0.16519441857691397 valid acc 15/16
Epoch 15 loss 0.4913757717055644 valid acc 15/16
Epoch 15 loss 0.3162886324307565 valid acc 15/16
Epoch 15 loss 0.4229762651786881 valid acc 15/16
Epoch 15 loss 0.37277303163126285 valid acc 15/16
Epoch 15 loss 0.2045128160107983 valid acc 15/16
Epoch 15 loss 0.1250857576105041 valid acc 15/16
Epoch 15 loss 0.5256605291092546 valid acc 15/16
Epoch 15 loss 0.20818926595246207 valid acc 15/16
Epoch 15 loss 0.3770724943276769 valid acc 15/16
Epoch 15 loss 0.3960346435879536 valid acc 15/16
Epoch 15 loss 0.5629294801905892 valid acc 14/16
Epoch 15 loss 0.052711700897510594 valid acc 14/16
Epoch 15 loss 0.9967685337302578 valid acc 15/16
Epoch 15 loss 0.4550509378676832 valid acc 15/16
Epoch 15 loss 0.09340576657104505 valid acc 15/16
Epoch 15 loss 0.19127064675766597 valid acc 15/16
Epoch 15 loss 0.42970356851552166 valid acc 14/16
Epoch 16 loss 0.003231844752808577 valid acc 14/16
Epoch 16 loss 0.43655807086093956 valid acc 15/16
Epoch 16 loss 0.4501545767554327 valid acc 15/16
Epoch 16 loss 0.4778533564302039 valid acc 15/16
Epoch 16 loss 0.06289850989623569 valid acc 15/16
Epoch 16 loss 0.4418712448119423 valid acc 15/16
Epoch 16 loss 0.24284004341007587 valid acc 15/16
Epoch 16 loss 0.6346169166555895 valid acc 15/16
Epoch 16 loss 0.16175496884130308 valid acc 15/16
Epoch 16 loss 0.4438322223282717 valid acc 15/16
Epoch 16 loss 0.1472389042333937 valid acc 15/16
Epoch 16 loss 0.6743916126981868 valid acc 15/16
Epoch 16 loss 0.46982631265266506 valid acc 15/16
Epoch 16 loss 0.45571496432423625 valid acc 15/16
Epoch 16 loss 0.8478146047057634 valid acc 15/16
Epoch 16 loss 0.9181064796573228 valid acc 15/16
Epoch 16 loss 0.6515956196728108 valid acc 15/16
Epoch 16 loss 0.2692552806580264 valid acc 15/16
Epoch 16 loss 0.5138951980536377 valid acc 15/16
Epoch 16 loss 0.8355958300189689 valid acc 15/16
Epoch 16 loss 0.9273081575787536 valid acc 15/16
Epoch 16 loss 0.5700901228150643 valid acc 15/16
Epoch 16 loss 0.11733961837904461 valid acc 15/16
Epoch 16 loss 1.0203817614600212 valid acc 15/16
Epoch 16 loss 0.37810586489197784 valid acc 15/16
Epoch 16 loss 0.41385537355041907 valid acc 15/16
Epoch 16 loss 0.31085616465041493 valid acc 15/16
Epoch 16 loss 0.20305717316057292 valid acc 15/16
Epoch 16 loss 0.31014088121779565 valid acc 15/16
Epoch 16 loss 0.20850842650294463 valid acc 15/16
Epoch 16 loss 0.8111646727902591 valid acc 14/16
Epoch 16 loss 0.575185680687623 valid acc 14/16
Epoch 16 loss 0.3432043013786636 valid acc 14/16
Epoch 16 loss 0.4221532501787441 valid acc 15/16
Epoch 16 loss 0.9889419426887587 valid acc 15/16
Epoch 16 loss 0.45283398787899687 valid acc 16/16
Epoch 16 loss 0.4053674812125989 valid acc 15/16
Epoch 16 loss 0.2870033177169063 valid acc 16/16
Epoch 16 loss 0.4376123806888042 valid acc 15/16
Epoch 16 loss 0.22587316969706134 valid acc 16/16
Epoch 16 loss 0.38494962752817213 valid acc 15/16
Epoch 16 loss 0.38183915750561315 valid acc 16/16
Epoch 16 loss 0.46042538468723915 valid acc 15/16
Epoch 16 loss 0.24472838243063688 valid acc 15/16
Epoch 16 loss 0.6243877834861222 valid acc 15/16
Epoch 16 loss 0.1776506245257381 valid acc 15/16
Epoch 16 loss 0.35304409066259745 valid acc 15/16
Epoch 16 loss 0.12501833665880263 valid acc 15/16
Epoch 16 loss 1.1094766681835115 valid acc 15/16
Epoch 16 loss 0.5003033406084796 valid acc 15/16
Epoch 16 loss 0.25430047380353016 valid acc 15/16
Epoch 16 loss 0.25669674764143213 valid acc 15/16
Epoch 16 loss 0.859737199286266 valid acc 15/16
Epoch 16 loss 0.36784117831656943 valid acc 15/16
Epoch 16 loss 0.5740846289338806 valid acc 14/16
Epoch 16 loss 0.6001918673127975 valid acc 15/16
Epoch 16 loss 0.25678331209277044 valid acc 14/16
Epoch 16 loss 0.40755114175365875 valid acc 14/16
Epoch 16 loss 0.43931077371253824 valid acc 14/16
Epoch 16 loss 0.1956101924154821 valid acc 14/16
Epoch 16 loss 0.11422435023378369 valid acc 14/16
Epoch 16 loss 0.6623712931871527 valid acc 15/16
Epoch 16 loss 0.48760010284251193 valid acc 15/16
Epoch 17 loss 0.026984068440593978 valid acc 15/16
Epoch 17 loss 0.5559996267039758 valid acc 15/16
Epoch 17 loss 0.2852222168480142 valid acc 15/16
Epoch 17 loss 0.3253688988067105 valid acc 15/16
Epoch 17 loss 0.48698639923266507 valid acc 15/16
Epoch 17 loss 0.5352405333325402 valid acc 15/16
Epoch 17 loss 0.38106926216913756 valid acc 15/16
Epoch 17 loss 0.5402538731271183 valid acc 15/16
Epoch 17 loss 0.22251634915869345 valid acc 15/16
Epoch 17 loss 0.2955011253258102 valid acc 15/16
Epoch 17 loss 0.20814866954769876 valid acc 15/16
Epoch 17 loss 0.5558793923341092 valid acc 15/16
Epoch 17 loss 0.36298011272110564 valid acc 15/16
Epoch 17 loss 0.3502462316489813 valid acc 15/16
Epoch 17 loss 0.7591597909036385 valid acc 15/16
Epoch 17 loss 0.4415577290963356 valid acc 15/16
Epoch 17 loss 0.7681136208263474 valid acc 15/16
Epoch 17 loss 0.5507880736873279 valid acc 15/16
Epoch 17 loss 0.7893207537688937 valid acc 15/16
Epoch 17 loss 0.278106939222932 valid acc 15/16
Epoch 17 loss 0.7955562790065702 valid acc 14/16
Epoch 17 loss 0.2687115240612912 valid acc 15/16
Epoch 17 loss 0.10184023070282944 valid acc 15/16
Epoch 17 loss 0.22604550810333296 valid acc 15/16
Epoch 17 loss 0.6439752237283582 valid acc 15/16
Epoch 17 loss 0.7246775694826827 valid acc 15/16
Epoch 17 loss 0.18945154045039836 valid acc 15/16
Epoch 17 loss 0.2829696717874659 valid acc 15/16
Epoch 17 loss 0.49886385067924965 valid acc 15/16
Epoch 17 loss 0.4135846974788173 valid acc 15/16
Epoch 17 loss 0.43890470957213895 valid acc 15/16
Epoch 17 loss 0.4255340178714391 valid acc 14/16
Epoch 17 loss 0.34797095696973185 valid acc 14/16
Epoch 17 loss 0.4102397716234893 valid acc 15/16
Epoch 17 loss 0.9099927623085803 valid acc 15/16
Epoch 17 loss 0.5005491414686704 valid acc 15/16
Epoch 17 loss 0.25834074834716314 valid acc 15/16
Epoch 17 loss 0.5035948041465013 valid acc 15/16
Epoch 17 loss 0.3209929499297044 valid acc 15/16
Epoch 17 loss 0.4303107645553408 valid acc 15/16
Epoch 17 loss 0.14865801157304598 valid acc 15/16
Epoch 17 loss 0.20741740294702213 valid acc 15/16
Epoch 17 loss 0.3352043676508826 valid acc 15/16
Epoch 17 loss 0.1670982435509411 valid acc 15/16
Epoch 17 loss 0.2857474723703224 valid acc 15/16
Epoch 17 loss 0.1564044006252907 valid acc 15/16
Epoch 17 loss 0.35383725327406906 valid acc 15/16
Epoch 17 loss 0.5393080727420292 valid acc 15/16
Epoch 17 loss 0.3914568933360121 valid acc 15/16
Epoch 17 loss 0.12106403721155068 valid acc 15/16
Epoch 17 loss 0.6455376810316606 valid acc 15/16
Epoch 17 loss 0.1908021113599802 valid acc 15/16
Epoch 17 loss 0.37111520241745455 valid acc 15/16
Epoch 17 loss 0.14783836799407346 valid acc 15/16
Epoch 17 loss 0.35945734443628785 valid acc 15/16
Epoch 17 loss 0.10618226188141067 valid acc 15/16
Epoch 17 loss 0.4527046141578689 valid acc 14/16
Epoch 17 loss 0.17522623729338718 valid acc 15/16
Epoch 17 loss 0.5757498642126273 valid acc 15/16
Epoch 17 loss 0.467664158195794 valid acc 15/16
Epoch 17 loss 0.13375939635439066 valid acc 14/16
Epoch 17 loss 0.15579812257383815 valid acc 14/16
Epoch 17 loss 0.13777148702622338 valid acc 14/16
Epoch 18 loss 0.0013122493256461087 valid acc 14/16
Epoch 18 loss 0.4058853362659558 valid acc 14/16
Epoch 18 loss 0.5736570532403238 valid acc 15/16
Epoch 18 loss 0.33124409991736836 valid acc 15/16
Epoch 18 loss 0.04221278731880507 valid acc 15/16
Epoch 18 loss 0.10672646723094165 valid acc 15/16
Epoch 18 loss 0.4526063585503274 valid acc 15/16
Epoch 18 loss 0.44928985267371885 valid acc 15/16
Epoch 18 loss 0.3959868568566341 valid acc 15/16
Epoch 18 loss 0.36779751358586055 valid acc 15/16
Epoch 18 loss 0.19906888564076763 valid acc 15/16
Epoch 18 loss 0.34516402518216693 valid acc 15/16
Epoch 18 loss 0.1479005611284916 valid acc 15/16
Epoch 18 loss 0.9192459960796455 valid acc 15/16
Epoch 18 loss 0.6114888310591916 valid acc 15/16
Epoch 18 loss 0.6055604808689282 valid acc 15/16
Epoch 18 loss 0.3725916462216995 valid acc 15/16
Epoch 18 loss 0.7338071078965668 valid acc 15/16
Epoch 18 loss 0.29800069907031185 valid acc 15/16
Epoch 18 loss 0.1231240848755096 valid acc 15/16
Epoch 18 loss 0.49699888832345457 valid acc 15/16
Epoch 18 loss 0.4921637166100664 valid acc 15/16
Epoch 18 loss 0.0849245483065515 valid acc 15/16
Epoch 18 loss 0.4736301471170545 valid acc 15/16
Epoch 18 loss 0.622408023967727 valid acc 15/16
Epoch 18 loss 0.4020846851773159 valid acc 15/16
Epoch 18 loss 0.1136812427515308 valid acc 15/16
Epoch 18 loss 0.14961922091967272 valid acc 15/16
Epoch 18 loss 0.4219159650955125 valid acc 15/16
Epoch 18 loss 0.10185934898467125 valid acc 15/16
Epoch 18 loss 0.1991771711143281 valid acc 15/16
Epoch 18 loss 0.5790407045193207 valid acc 14/16
Epoch 18 loss 0.06577391635589491 valid acc 14/16
Epoch 18 loss 0.3322214184901478 valid acc 14/16
Epoch 18 loss 1.0603883146677138 valid acc 14/16
Epoch 18 loss 0.8242477161970305 valid acc 15/16
Epoch 18 loss 0.4242605006402799 valid acc 16/16
Epoch 18 loss 0.3960777826941316 valid acc 16/16
Epoch 18 loss 0.49973790997264284 valid acc 15/16
Epoch 18 loss 0.5126115736890319 valid acc 15/16
Epoch 18 loss 0.36227405629665704 valid acc 15/16
Epoch 18 loss 0.37713145203568976 valid acc 15/16
Epoch 18 loss 0.1578474627090131 valid acc 15/16
Epoch 18 loss 0.35068763162688094 valid acc 15/16
Epoch 18 loss 0.5321144077349369 valid acc 16/16
Epoch 18 loss 0.13004491386252703 valid acc 16/16
Epoch 18 loss 0.5496994882913996 valid acc 15/16
Epoch 18 loss 0.6560315404899677 valid acc 15/16
Epoch 18 loss 0.31701553730724297 valid acc 15/16
Epoch 18 loss 0.05212299791473851 valid acc 15/16
Epoch 18 loss 0.1480926559497021 valid acc 15/16
Epoch 18 loss 0.30875702016652457 valid acc 15/16
Epoch 18 loss 0.48832409983561503 valid acc 15/16
Epoch 18 loss 0.2848672210779992 valid acc 15/16
Epoch 18 loss 0.21185432247713534 valid acc 15/16
Epoch 18 loss 0.24396196437999657 valid acc 15/16
Epoch 18 loss 0.3566401983884989 valid acc 15/16
Epoch 18 loss 0.4573741639765439 valid acc 15/16
Epoch 18 loss 0.8412182734260084 valid acc 15/16
Epoch 18 loss 0.7376246270669473 valid acc 15/16
Epoch 18 loss 0.4820961483554308 valid acc 15/16
Epoch 18 loss 0.21517256947970947 valid acc 15/16
Epoch 18 loss 0.18868366824935773 valid acc 15/16
Epoch 19 loss 0.002217320962311309 valid acc 15/16
Epoch 19 loss 0.12618669352679526 valid acc 15/16
Epoch 19 loss 0.4621383317845287 valid acc 15/16
Epoch 19 loss 0.21333541985898202 valid acc 15/16
Epoch 19 loss 0.1827597346396923 valid acc 15/16
Epoch 19 loss 0.2190671045410416 valid acc 15/16
Epoch 19 loss 0.3202192978485619 valid acc 15/16
Epoch 19 loss 1.080860495997336 valid acc 15/16
Epoch 19 loss 0.1640418790501582 valid acc 15/16
Epoch 19 loss 0.11556892524779272 valid acc 15/16
Epoch 19 loss 0.2753652033552797 valid acc 15/16
Epoch 19 loss 0.4122135731922843 valid acc 15/16
Epoch 19 loss 0.39378627444266096 valid acc 15/16
Epoch 19 loss 0.22248268692373058 valid acc 15/16
Epoch 19 loss 0.40128905868355963 valid acc 15/16
Epoch 19 loss 0.3725664479470622 valid acc 15/16
Epoch 19 loss 0.9294082485824575 valid acc 15/16
Epoch 19 loss 0.21932662611438752 valid acc 15/16
Epoch 19 loss 0.8998197873748984 valid acc 15/16
Epoch 19 loss 0.20718401611845547 valid acc 15/16
Epoch 19 loss 0.6452212166795558 valid acc 15/16
Epoch 19 loss 0.4543712973312754 valid acc 15/16
Epoch 19 loss 0.16698781682818237 valid acc 15/16
Epoch 19 loss 0.5324794433266384 valid acc 14/16
Epoch 19 loss 0.3500069750672426 valid acc 15/16
Epoch 19 loss 0.2829432793095776 valid acc 15/16
Epoch 19 loss 0.07561528500765136 valid acc 15/16
Epoch 19 loss 0.1923702092452214 valid acc 15/16
Epoch 19 loss 0.3093073130379315 valid acc 15/16
Epoch 19 loss 0.1413206918121769 valid acc 15/16
Epoch 19 loss 0.4636499399653554 valid acc 15/16
Epoch 19 loss 0.5376648490677343 valid acc 14/16
Epoch 19 loss 0.14832630196392466 valid acc 15/16
Epoch 19 loss 0.17462258121383345 valid acc 15/16
Epoch 19 loss 0.7710958384518685 valid acc 14/16
Epoch 19 loss 0.34190180413844184 valid acc 15/16
Epoch 19 loss 0.37142566134749694 valid acc 16/16
Epoch 19 loss 0.20377847677820998 valid acc 16/16
Epoch 19 loss 0.2587821569339282 valid acc 16/16
Epoch 19 loss 0.41691625217869227 valid acc 16/16
Epoch 19 loss 0.23447298615940398 valid acc 16/16
Epoch 19 loss 0.1697544768976909 valid acc 16/16
Epoch 19 loss 0.4476443899144507 valid acc 15/16
Epoch 19 loss 0.36632105087326616 valid acc 16/16
Epoch 19 loss 0.1402144664417162 valid acc 16/16
Epoch 19 loss 0.19790344342102756 valid acc 16/16
Epoch 19 loss 0.3529643228230265 valid acc 16/16
Epoch 19 loss 0.5237966635971136 valid acc 15/16
Epoch 19 loss 0.3245988387085422 valid acc 16/16
Epoch 19 loss 0.2030438729375389 valid acc 16/16
Epoch 19 loss 0.172688349608495 valid acc 15/16
Epoch 19 loss 0.17712950238575267 valid acc 15/16
Epoch 19 loss 0.15940579654359188 valid acc 15/16
Epoch 19 loss 0.11103300631761143 valid acc 15/16
Epoch 19 loss 0.39074652857478676 valid acc 15/16
Epoch 19 loss 0.15008638144247566 valid acc 15/16
Epoch 19 loss 0.31666283610019014 valid acc 15/16
Epoch 19 loss 0.16174580045838194 valid acc 15/16
Epoch 19 loss 0.4355210696077502 valid acc 15/16
Epoch 19 loss 0.35206571925977936 valid acc 14/16
Epoch 19 loss 0.1571827978590023 valid acc 14/16
Epoch 19 loss 0.28500795203855994 valid acc 15/16
Epoch 19 loss 0.13548929020922323 valid acc 14/16
Epoch 20 loss 0.0028533664906886247 valid acc 15/16
Epoch 20 loss 0.34276722531757214 valid acc 15/16
Epoch 20 loss 0.284348703723512 valid acc 15/16
Epoch 20 loss 0.38828567643339046 valid acc 15/16
Epoch 20 loss 0.2177025673876108 valid acc 15/16
Epoch 20 loss 0.44870256070079045 valid acc 15/16
Epoch 20 loss 1.048086498994886 valid acc 15/16
Epoch 20 loss 0.5133334191234604 valid acc 15/16
Epoch 20 loss 0.3788565473390899 valid acc 15/16
Epoch 20 loss 0.18346513324445302 valid acc 15/16
Epoch 20 loss 0.06411131980598422 valid acc 15/16
Epoch 20 loss 0.4468891998817499 valid acc 15/16
Epoch 20 loss 0.7669833519559908 valid acc 15/16
Epoch 20 loss 0.5562270323668523 valid acc 15/16
Epoch 20 loss 0.7485394102809068 valid acc 15/16
Epoch 20 loss 0.42471233283918397 valid acc 15/16
Epoch 20 loss 0.4256509906657681 valid acc 15/16
Epoch 20 loss 0.5139439344398391 valid acc 15/16
Epoch 20 loss 0.4943756119408449 valid acc 15/16
Epoch 20 loss 0.4250547000985293 valid acc 15/16
Epoch 20 loss 0.6194354620517184 valid acc 15/16
Epoch 20 loss 0.39745359956712684 valid acc 15/16
Epoch 20 loss 0.14364662933359273 valid acc 15/16
Epoch 20 loss 0.15858423731892862 valid acc 14/16
Epoch 20 loss 0.2626372910279741 valid acc 15/16
Epoch 20 loss 0.4630035341716794 valid acc 15/16
Epoch 20 loss 0.22017510668093487 valid acc 15/16
Epoch 20 loss 0.060637089676459255 valid acc 15/16
Epoch 20 loss 0.2074297274455849 valid acc 15/16
Epoch 20 loss 0.2548443525965495 valid acc 15/16
Epoch 20 loss 0.21021946881530124 valid acc 15/16
Epoch 20 loss 0.6707147178605066 valid acc 14/16
Epoch 20 loss 0.23167310695938725 valid acc 14/16
Epoch 20 loss 0.3032577115292449 valid acc 15/16
Epoch 20 loss 0.7994099312743719 valid acc 15/16
Epoch 20 loss 0.14419980966580198 valid acc 15/16
Epoch 20 loss 0.19624724822222778 valid acc 15/16
Epoch 20 loss 0.20838408733769048 valid acc 15/16
Epoch 20 loss 0.34557720732975494 valid acc 15/16
Epoch 20 loss 0.11481463835252931 valid acc 16/16
Epoch 20 loss 0.12737291221842006 valid acc 15/16
Epoch 20 loss 0.40171054923263644 valid acc 15/16
Epoch 20 loss 0.379038041571989 valid acc 15/16
Epoch 20 loss 0.15597828921978824 valid acc 15/16
Epoch 20 loss 0.530497772323788 valid acc 16/16
Epoch 20 loss 0.1703387478058192 valid acc 16/16
Epoch 20 loss 0.35313669897812555 valid acc 15/16
Epoch 20 loss 0.6669588793205563 valid acc 15/16
Epoch 20 loss 0.18803344732899382 valid acc 15/16
Epoch 20 loss 0.13406515466102542 valid acc 15/16
Epoch 20 loss 0.33408618265669177 valid acc 15/16
Epoch 20 loss 0.2791199991419781 valid acc 15/16
Epoch 20 loss 0.19938259275480197 valid acc 16/16
Epoch 20 loss 0.07389704459735263 valid acc 16/16
Epoch 20 loss 0.3135228483541044 valid acc 15/16
Epoch 20 loss 0.5878035065552211 valid acc 15/16
Epoch 20 loss 0.5012753805159584 valid acc 14/16
Epoch 20 loss 0.4630510531835481 valid acc 14/16
Epoch 20 loss 0.5165762707417502 valid acc 15/16
Epoch 20 loss 0.5027135033893357 valid acc 14/16
Epoch 20 loss 0.1648116502147881 valid acc 15/16
Epoch 20 loss 0.13802602957979387 valid acc 15/16
Epoch 20 loss 0.061220998870469445 valid acc 15/16
Epoch 21 loss 0.0032827535587699463 valid acc 15/16
Epoch 21 loss 0.2895260129226259 valid acc 15/16
Epoch 21 loss 0.2407682124078428 valid acc 15/16
Epoch 21 loss 0.35721177371344637 valid acc 15/16
Epoch 21 loss 0.36923148314106347 valid acc 15/16
Epoch 21 loss 0.20475967798629874 valid acc 15/16
Epoch 21 loss 0.4151881451185328 valid acc 15/16
Epoch 21 loss 0.277099835515387 valid acc 15/16
Epoch 21 loss 0.1464701068611653 valid acc 15/16
Epoch 21 loss 0.05584338067081962 valid acc 15/16
Epoch 21 loss 0.13430669064411394 valid acc 15/16
Epoch 21 loss 0.3180142659523283 valid acc 15/16
Epoch 21 loss 0.3660463489855056 valid acc 15/16
Epoch 21 loss 0.24967602295467206 valid acc 15/16
Epoch 21 loss 0.31514271037264413 valid acc 15/16
Epoch 21 loss 0.4622271449385523 valid acc 15/16
Epoch 21 loss 0.3352048129842943 valid acc 15/16
Epoch 21 loss 0.5167005247597642 valid acc 14/16
Epoch 21 loss 0.3242604344009956 valid acc 15/16
Epoch 21 loss 0.08270272177210242 valid acc 15/16
Epoch 21 loss 0.7728286877075443 valid acc 15/16
Epoch 21 loss 0.41092967051415796 valid acc 16/16
Epoch 21 loss 0.12419719854121364 valid acc 15/16
Epoch 21 loss 0.14986035528578107 valid acc 15/16
Epoch 21 loss 0.25371038766580367 valid acc 15/16
Epoch 21 loss 0.28098908274788864 valid acc 15/16
Epoch 21 loss 0.20818334874772093 valid acc 15/16
Epoch 21 loss 0.06740618729368508 valid acc 15/16
Epoch 21 loss 0.1838627748850808 valid acc 15/16
Epoch 21 loss 0.37034990823446107 valid acc 15/16
Epoch 21 loss 0.23786775972918195 valid acc 15/16
Epoch 21 loss 0.23146354452714318 valid acc 14/16
Epoch 21 loss 0.26086672350127393 valid acc 15/16
Epoch 21 loss 0.37256223677605654 valid acc 15/16
Epoch 21 loss 0.8844319991863743 valid acc 15/16
Epoch 21 loss 0.2330471679466472 valid acc 15/16
Epoch 21 loss 0.5322530266117352 valid acc 16/16
Epoch 21 loss 0.3120235572182761 valid acc 16/16
Epoch 21 loss 0.19168325028594912 valid acc 16/16
Epoch 21 loss 0.2120294877828445 valid acc 16/16
Epoch 21 loss 0.1846452762400813 valid acc 15/16
Epoch 21 loss 0.23004636354703836 valid acc 15/16
Epoch 21 loss 0.057734219471637815 valid acc 15/16
Epoch 21 loss 0.11747834018992703 valid acc 15/16
Epoch 21 loss 0.4150469473623438 valid acc 16/16
Epoch 21 loss 0.06777050806691148 valid acc 16/16
Epoch 21 loss 0.21522356533501513 valid acc 16/16
Epoch 21 loss 0.5374112926653174 valid acc 15/16
Epoch 21 loss 0.12948446226088323 valid acc 15/16
Epoch 21 loss 0.06660082208524093 valid acc 15/16
Epoch 21 loss 0.36803287092371667 valid acc 16/16
Epoch 21 loss 0.5540910864170254 valid acc 16/16
Epoch 21 loss 0.5040427478366654 valid acc 15/16
Epoch 21 loss 0.1263338864133317 valid acc 15/16
Epoch 21 loss 0.37659343418113994 valid acc 15/16
Epoch 21 loss 0.15110419326946795 valid acc 15/16
Epoch 21 loss 0.29983943189387174 valid acc 15/16
Epoch 21 loss 0.057551631259136315 valid acc 15/16
Epoch 21 loss 0.14581735558083542 valid acc 15/16
Epoch 21 loss 0.31390131294161344 valid acc 15/16
Epoch 21 loss 0.11535202225928853 valid acc 15/16
Epoch 21 loss 0.2467186099920119 valid acc 16/16
Epoch 21 loss 0.543567459377658 valid acc 14/16
Epoch 22 loss 0.003564415066660609 valid acc 14/16
Epoch 22 loss 0.3486945857350501 valid acc 15/16
Epoch 22 loss 0.4696816681538848 valid acc 15/16
Epoch 22 loss 0.5353708712731928 valid acc 15/16
Epoch 22 loss 0.1267795683198571 valid acc 15/16
Epoch 22 loss 0.23323194286114446 valid acc 15/16
Epoch 22 loss 0.5774628912698923 valid acc 15/16
Epoch 22 loss 0.306810026827002 valid acc 15/16
Epoch 22 loss 0.17584137968816493 valid acc 15/16
Epoch 22 loss 0.30709471108612146 valid acc 15/16
Epoch 22 loss 0.12707899360490604 valid acc 15/16
Epoch 22 loss 0.22169904378765387 valid acc 15/16
Epoch 22 loss 0.10049393640198381 valid acc 15/16
Epoch 22 loss 0.49983115613464446 valid acc 15/16
Epoch 22 loss 0.2542253664610854 valid acc 15/16
Epoch 22 loss 0.39579732474631896 valid acc 15/16
Epoch 22 loss 0.18374849139930477 valid acc 15/16
Epoch 22 loss 0.560440244570914 valid acc 15/16
Epoch 22 loss 0.2989036985964873 valid acc 15/16
Epoch 22 loss 0.07618322664753407 valid acc 15/16
Epoch 22 loss 0.4708665684263094 valid acc 16/16
Epoch 22 loss 0.1554238069410923 valid acc 16/16
Epoch 22 loss 0.07042575003269125 valid acc 16/16
Epoch 22 loss 0.16327615274966173 valid acc 15/16
Epoch 22 loss 0.5934326931773495 valid acc 14/16
Epoch 22 loss 0.35768590144805373 valid acc 15/16
Epoch 22 loss 0.1312562299013981 valid acc 15/16
Epoch 22 loss 0.06336968457571501 valid acc 15/16
Epoch 22 loss 0.19656090884553468 valid acc 15/16
Epoch 22 loss 0.021761258272444295 valid acc 15/16
Epoch 22 loss 0.14387972777214764 valid acc 15/16
Epoch 22 loss 0.10521969307162227 valid acc 15/16
Epoch 22 loss 0.5900462626297641 valid acc 15/16
Epoch 22 loss 0.4498084326825312 valid acc 15/16
Epoch 22 loss 0.6090507163090358 valid acc 15/16
Epoch 22 loss 0.45511124738148984 valid acc 15/16
Epoch 22 loss 0.296827813169298 valid acc 15/16
Epoch 22 loss 0.5953148196718263 valid acc 15/16
Epoch 22 loss 0.5039107497827204 valid acc 15/16
Epoch 22 loss 0.1977252525473719 valid acc 15/16
Epoch 22 loss 0.10950384543764546 valid acc 15/16
Epoch 22 loss 0.16012539205672327 valid acc 15/16
Epoch 22 loss 0.0703895265640675 valid acc 15/16
Epoch 22 loss 0.10029977830214695 valid acc 15/16
Epoch 22 loss 0.1300677539552485 valid acc 15/16
Epoch 22 loss 0.0406474037189567 valid acc 15/16
Epoch 22 loss 0.13735205615131318 valid acc 15/16
Epoch 22 loss 0.505597294976289 valid acc 15/16
Epoch 22 loss 0.27057943998248546 valid acc 15/16
Epoch 22 loss 0.12660218644413018 valid acc 15/16
Epoch 22 loss 0.09670386866959924 valid acc 15/16
Epoch 22 loss 0.21250674769369743 valid acc 15/16
Epoch 22 loss 0.29990790258037064 valid acc 15/16
Epoch 22 loss 0.18074255438522485 valid acc 15/16
Epoch 22 loss 0.35678531772182737 valid acc 15/16
Epoch 22 loss 0.18414034405419716 valid acc 15/16
Epoch 22 loss 0.4336693999950466 valid acc 15/16
Epoch 22 loss 0.07041596288952012 valid acc 15/16
Epoch 22 loss 0.3723316783688406 valid acc 15/16
Epoch 22 loss 0.3236752437275526 valid acc 15/16
Epoch 22 loss 0.8261027291903837 valid acc 15/16
Epoch 22 loss 0.7771416872673141 valid acc 16/16
Epoch 22 loss 0.468086442780174 valid acc 15/16
Epoch 23 loss 0.011790257429438955 valid acc 15/16
Epoch 23 loss 0.5929640228439182 valid acc 15/16
Epoch 23 loss 0.6346398081599206 valid acc 15/16
Epoch 23 loss 0.5128917760935872 valid acc 15/16
Epoch 23 loss 0.08253068557219118 valid acc 15/16
Epoch 23 loss 0.2291538963680489 valid acc 15/16
Epoch 23 loss 0.4558422294061474 valid acc 15/16
Epoch 23 loss 0.3686323650650313 valid acc 15/16
Epoch 23 loss 0.3509482378546557 valid acc 15/16
Epoch 23 loss 0.22806176056780275 valid acc 15/16
Epoch 23 loss 0.20696805807062474 valid acc 15/16
Epoch 23 loss 0.5351876085281225 valid acc 15/16
Epoch 23 loss 0.09448734480151799 valid acc 15/16
Epoch 23 loss 0.424335693896622 valid acc 15/16
Epoch 23 loss 0.34789235602212565 valid acc 15/16
Epoch 23 loss 0.18964090895800678 valid acc 15/16
Epoch 23 loss 0.31522613183661413 valid acc 15/16
Epoch 23 loss 0.49836505205530524 valid acc 15/16
Epoch 23 loss 0.7262148014624616 valid acc 15/16
Epoch 23 loss 0.13433814852718928 valid acc 15/16
Epoch 23 loss 0.7224522514367058 valid acc 15/16
Epoch 23 loss 0.08154009405496526 valid acc 15/16
Epoch 23 loss 0.3825892370368276 valid acc 15/16
Epoch 23 loss 0.2609972188234432 valid acc 15/16
Epoch 23 loss 0.5410369529240132 valid acc 15/16
Epoch 23 loss 0.2999248602945223 valid acc 15/16
Epoch 23 loss 0.1453759053542089 valid acc 15/16
Epoch 23 loss 0.2425444716171987 valid acc 15/16
Epoch 23 loss 0.14643894127802107 valid acc 15/16
Epoch 23 loss 0.13288185004313352 valid acc 15/16
Epoch 23 loss 0.06617716590310058 valid acc 15/16
Epoch 23 loss 0.2573341026674576 valid acc 15/16
Epoch 23 loss 0.15392015494893757 valid acc 16/16
Epoch 23 loss 0.19147899462714613 valid acc 15/16
Epoch 23 loss 0.9054866933802811 valid acc 15/16
Epoch 23 loss 0.2839327366491995 valid acc 15/16
Epoch 23 loss 0.10204351454214239 valid acc 15/16
Epoch 23 loss 0.32506893261222625 valid acc 16/16
Epoch 23 loss 0.3742578000886393 valid acc 16/16
Epoch 23 loss 0.20191662188296522 valid acc 16/16
Epoch 23 loss 0.3060045354961277 valid acc 16/16
Epoch 23 loss 0.1677745742826206 valid acc 16/16
Epoch 23 loss 0.4331092606049124 valid acc 16/16
Epoch 23 loss 0.3574247170752294 valid acc 16/16
Epoch 23 loss 0.32242543847867144 valid acc 14/16
Epoch 23 loss 0.252126528478361 valid acc 15/16
Epoch 23 loss 0.3307139106083951 valid acc 15/16
Epoch 23 loss 0.3991839408048268 valid acc 15/16
Epoch 23 loss 0.07277224749727301 valid acc 15/16
Epoch 23 loss 0.35357353186241336 valid acc 15/16
Epoch 23 loss 0.1906609773238362 valid acc 15/16
Epoch 23 loss 0.30152285977200943 valid acc 15/16
Epoch 23 loss 0.35698729687948294 valid acc 15/16
Epoch 23 loss 0.32177311729537666 valid acc 15/16
Epoch 23 loss 0.19490670124266646 valid acc 15/16
Epoch 23 loss 0.2542903922406663 valid acc 15/16
Epoch 23 loss 0.4738800130184636 valid acc 15/16
Epoch 23 loss 0.0401579014851014 valid acc 15/16
Epoch 23 loss 0.3988297104309272 valid acc 15/16
Epoch 23 loss 0.20150912094270246 valid acc 15/16
Epoch 23 loss 0.00877365943524272 valid acc 15/16
Epoch 23 loss 0.1454806646983795 valid acc 15/16
Epoch 23 loss 0.5602459316932263 valid acc 15/16
Epoch 24 loss 0.0002091489516828915 valid acc 15/16
Epoch 24 loss 0.6281501594050176 valid acc 15/16
Epoch 24 loss 0.46909936574503625 valid acc 15/16
Epoch 24 loss 0.3956040093006588 valid acc 15/16
Epoch 24 loss 0.08022267759708729 valid acc 15/16
Epoch 24 loss 0.2565511164475446 valid acc 15/16
Epoch 24 loss 0.4519021860867021 valid acc 15/16
Epoch 24 loss 0.3814791047113312 valid acc 15/16
Epoch 24 loss 0.5437955022994093 valid acc 15/16
Epoch 24 loss 0.10615417982749836 valid acc 15/16
Epoch 24 loss 0.2838726238092564 valid acc 15/16
Epoch 24 loss 0.12881726142239708 valid acc 15/16
Epoch 24 loss 0.23832465944896622 valid acc 15/16
Epoch 24 loss 0.3160510770757937 valid acc 15/16
Epoch 24 loss 0.46980452776487686 valid acc 15/16
Epoch 24 loss 0.1616870940515614 valid acc 15/16
Epoch 24 loss 0.3408066077821225 valid acc 15/16
Epoch 24 loss 0.15765004078637773 valid acc 15/16
Epoch 24 loss 0.35799716816592037 valid acc 15/16
Epoch 24 loss 0.09172124570126217 valid acc 15/16
Epoch 24 loss 0.5008306975086991 valid acc 14/16
Epoch 24 loss 0.12773534378608364 valid acc 14/16
Epoch 24 loss 0.038848704411535984 valid acc 14/16
Epoch 24 loss 0.22619650169089156 valid acc 14/16
Epoch 24 loss 0.09174512612452795 valid acc 14/16
Epoch 24 loss 0.528565337588077 valid acc 14/16
Epoch 24 loss 0.18308318871042084 valid acc 14/16
Epoch 24 loss 0.26293784359534755 valid acc 15/16
Epoch 24 loss 0.18344675972096525 valid acc 15/16
Epoch 24 loss 0.1092340329773614 valid acc 14/16
Epoch 24 loss 0.153389707008216 valid acc 14/16
Epoch 24 loss 0.16825702510214113 valid acc 14/16
Epoch 24 loss 0.022319468349957794 valid acc 14/16
Epoch 24 loss 0.29876592704133115 valid acc 14/16
Epoch 24 loss 0.6559564616420552 valid acc 15/16
Epoch 24 loss 0.4752827521732848 valid acc 15/16
Epoch 24 loss 0.28028037376656256 valid acc 16/16
Epoch 24 loss 0.15424344092293596 valid acc 15/16
Epoch 24 loss 0.293194752383997 valid acc 15/16
Epoch 24 loss 0.12821316861490156 valid acc 15/16
Epoch 24 loss 0.0570254330216432 valid acc 15/16
Epoch 24 loss 0.33127787290574123 valid acc 16/16
Epoch 24 loss 0.48767613350094197 valid acc 16/16
Epoch 24 loss 0.05066046848940975 valid acc 16/16
Epoch 24 loss 0.4679299418335575 valid acc 16/16
Epoch 24 loss 0.09802559299714572 valid acc 16/16
Epoch 24 loss 0.5013479298555816 valid acc 16/16
Epoch 24 loss 0.3374757058651918 valid acc 15/16
Epoch 24 loss 0.31944858735242093 valid acc 15/16
Epoch 24 loss 0.08732708810186268 valid acc 15/16
Epoch 24 loss 0.06465940547699783 valid acc 15/16
Epoch 24 loss 0.16837816620935692 valid acc 15/16
Epoch 24 loss 0.2211309258496078 valid acc 15/16
Epoch 24 loss 0.07182037778770034 valid acc 15/16
Epoch 24 loss 0.15485520595254693 valid acc 15/16
Epoch 24 loss 0.09474352602811315 valid acc 15/16
Epoch 24 loss 0.26383072362927656 valid acc 15/16
Epoch 24 loss 0.03793315209132003 valid acc 15/16
Epoch 24 loss 0.22973200628064477 valid acc 15/16
Epoch 24 loss 0.3955369136481154 valid acc 14/16
Epoch 24 loss 0.34332436772078867 valid acc 14/16
Epoch 24 loss 0.29013232671915384 valid acc 14/16
Epoch 24 loss 0.285499338217005 valid acc 14/16
Epoch 25 loss 0.0011908031357521498 valid acc 14/16
Epoch 25 loss 0.42166778266345045 valid acc 15/16
Epoch 25 loss 0.23498773650272486 valid acc 15/16
Epoch 25 loss 0.3970043656834247 valid acc 15/16
Epoch 25 loss 0.13398453448580194 valid acc 15/16
Epoch 25 loss 0.3891407580561541 valid acc 15/16
Epoch 25 loss 0.5832365566459838 valid acc 15/16
Epoch 25 loss 0.48939270836220206 valid acc 15/16
Epoch 25 loss 0.1807643538582644 valid acc 15/16
Epoch 25 loss 0.3487624270302031 valid acc 15/16
Epoch 25 loss 0.20704719358579488 valid acc 15/16
Epoch 25 loss 0.39620252638887876 valid acc 15/16
Epoch 25 loss 0.23961707926974823 valid acc 15/16
Epoch 25 loss 0.4238912927313867 valid acc 15/16
Epoch 25 loss 0.3739289676418347 valid acc 15/16
Epoch 25 loss 0.161576550525596 valid acc 15/16
Epoch 25 loss 1.112012351847545 valid acc 15/16
Epoch 25 loss 0.4846581165046591 valid acc 15/16
Epoch 25 loss 0.25448382103329265 valid acc 15/16
Epoch 25 loss 0.20509086748097016 valid acc 15/16
Epoch 25 loss 0.6444404476937251 valid acc 14/16
Epoch 25 loss 0.46348573266063514 valid acc 14/16
Epoch 25 loss 0.16531490778071167 valid acc 15/16
Epoch 25 loss 0.2531521087039458 valid acc 14/16
Epoch 25 loss 0.1487355422530825 valid acc 14/16
Epoch 25 loss 1.1394586602649552 valid acc 15/16
Epoch 25 loss 0.5036419652559015 valid acc 16/16
Epoch 25 loss 0.31830495407634407 valid acc 15/16
Epoch 25 loss 0.2465650556736224 valid acc 15/16
Epoch 25 loss 0.3325520497863929 valid acc 15/16
Epoch 25 loss 0.3264125671291927 valid acc 16/16
Epoch 25 loss 0.2166042224957072 valid acc 16/16
Epoch 25 loss 0.1656070187087933 valid acc 16/16
Epoch 25 loss 0.44721221358378815 valid acc 16/16
Epoch 25 loss 0.8527982910222051 valid acc 15/16
Epoch 25 loss 0.4635860325970711 valid acc 15/16
Epoch 25 loss 0.25407705044837076 valid acc 16/16
Epoch 25 loss 0.32422623247630755 valid acc 15/16
Epoch 25 loss 0.25089475870367167 valid acc 16/16
Epoch 25 loss 0.45866733378545776 valid acc 16/16
Epoch 25 loss 0.47460873677136417 valid acc 15/16
Epoch 25 loss 0.22833306939488623 valid acc 15/16
Epoch 25 loss 0.15548498443133324 valid acc 15/16
Epoch 25 loss 0.14442179274408165 valid acc 15/16
Epoch 25 loss 0.21042714750035274 valid acc 15/16
Epoch 25 loss 0.19424029027596326 valid acc 15/16
Epoch 25 loss 0.19803188724467813 valid acc 15/16
Epoch 25 loss 0.32325341937906 valid acc 15/16
Epoch 25 loss 0.1421489925782789 valid acc 15/16
Epoch 25 loss 0.16305591471311842 valid acc 15/16
Epoch 25 loss 0.46618185647190286 valid acc 15/16
Epoch 25 loss 0.24116730709147255 valid acc 16/16
Epoch 25 loss 0.1867527231323372 valid acc 15/16
Epoch 25 loss 0.10875744677083149 valid acc 15/16
Epoch 25 loss 0.16410923196453836 valid acc 15/16
Epoch 25 loss 0.1480701828585156 valid acc 15/16
Epoch 25 loss 0.5748825529525929 valid acc 15/16
Epoch 25 loss 0.09341950834413904 valid acc 15/16
Epoch 25 loss 0.28551934901773923 valid acc 15/16
Epoch 25 loss 0.2551882650705433 valid acc 15/16
Epoch 25 loss 0.0610187708593668 valid acc 15/16
Epoch 25 loss 0.14219458693464504 valid acc 15/16
Epoch 25 loss 0.07159109208739822 valid acc 15/16
Epoch 26 loss 0.17246817846936002 valid acc 15/16
Epoch 26 loss 0.17724264495961312 valid acc 15/16
Epoch 26 loss 0.2610265076019944 valid acc 15/16
Epoch 26 loss 0.14797207010823488 valid acc 15/16
Epoch 26 loss 0.07264445562476024 valid acc 15/16
Epoch 26 loss 0.04687171962519732 valid acc 15/16
Epoch 26 loss 0.2611819450208337 valid acc 15/16
Epoch 26 loss 0.45895869148055934 valid acc 15/16
Epoch 26 loss 0.15950937573293436 valid acc 15/16
Epoch 26 loss 0.10235431980869453 valid acc 15/16
Epoch 26 loss 0.07674080051287696 valid acc 15/16
Epoch 26 loss 0.5855013795604888 valid acc 15/16
Epoch 26 loss 0.3319082073675407 valid acc 15/16
Epoch 26 loss 0.40290556472694056 valid acc 15/16
Epoch 26 loss 0.5758982051411008 valid acc 15/16
Epoch 26 loss 0.051846494073091776 valid acc 15/16
Epoch 26 loss 0.8386317111566424 valid acc 15/16
Epoch 26 loss 0.18313548873706315 valid acc 15/16
Epoch 26 loss 0.22603540523298382 valid acc 15/16
Epoch 26 loss 0.22062324634347752 valid acc 15/16
Epoch 26 loss 0.6586684675846781 valid acc 15/16
Epoch 26 loss 0.3276401678696321 valid acc 15/16
Epoch 26 loss 0.02331146383238558 valid acc 15/16
Epoch 26 loss 0.18692606002459594 valid acc 15/16
Epoch 26 loss 0.30140526513112537 valid acc 15/16
Epoch 26 loss 0.3286171439623841 valid acc 15/16
Epoch 26 loss 0.12135038610307347 valid acc 15/16
Epoch 26 loss 0.3641538780784824 valid acc 15/16
Epoch 26 loss 0.24523874126855383 valid acc 15/16
Epoch 26 loss 0.09064820053661976 valid acc 15/16
Epoch 26 loss 0.3778294734818227 valid acc 15/16
Epoch 26 loss 0.3049389430041902 valid acc 15/16
Epoch 26 loss 0.030614199119791352 valid acc 15/16
Epoch 26 loss 0.08546770918659796 valid acc 15/16
Epoch 26 loss 0.6056052343192031 valid acc 15/16
Epoch 26 loss 0.44504170537950305 valid acc 15/16
Epoch 26 loss 0.1434250099933736 valid acc 16/16
Epoch 26 loss 0.04163477693768247 valid acc 16/16
Epoch 26 loss 0.3527845401554676 valid acc 16/16
Epoch 26 loss 0.10323118871248738 valid acc 16/16
Epoch 26 loss 0.09579515184325982 valid acc 16/16
Epoch 26 loss 0.25519626370427384 valid acc 16/16
Epoch 26 loss 0.2801491077682697 valid acc 15/16
Epoch 26 loss 0.030542540735163937 valid acc 15/16
Epoch 26 loss 0.19529675751433734 valid acc 15/16
Epoch 26 loss 0.06720773098601501 valid acc 15/16
Epoch 26 loss 0.5170510640888941 valid acc 15/16
Epoch 26 loss 0.6968411141115346 valid acc 15/16
Epoch 26 loss 0.15478272333991605 valid acc 15/16
Epoch 26 loss 0.1545798109208424 valid acc 15/16
Epoch 26 loss 0.3851805480819567 valid acc 15/16
Epoch 26 loss 0.3301341394749696 valid acc 15/16
Epoch 26 loss 0.7544366574400613 valid acc 15/16
Epoch 26 loss 0.4108840492802235 valid acc 15/16
Epoch 26 loss 0.20115092276015809 valid acc 15/16
Epoch 26 loss 0.22325856095039914 valid acc 15/16
Epoch 26 loss 0.14303186359878528 valid acc 15/16
Epoch 26 loss 0.3483704575768367 valid acc 15/16
Epoch 26 loss 0.3434897329737372 valid acc 15/16
Epoch 26 loss 0.41075630910743377 valid acc 15/16
Epoch 26 loss 0.42381871393404635 valid acc 15/16
Epoch 26 loss 0.15446513159885727 valid acc 15/16
Epoch 26 loss 0.20629679577475518 valid acc 15/16
Epoch 27 loss 0.002363402428877226 valid acc 16/16
Epoch 27 loss 0.22799446995029163 valid acc 15/16
Epoch 27 loss 0.1007479699286265 valid acc 15/16
Epoch 27 loss 0.2898525807398727 valid acc 15/16
Epoch 27 loss 0.07751728287126475 valid acc 15/16
Epoch 27 loss 0.6060335632974301 valid acc 15/16
Epoch 27 loss 0.3967291800468622 valid acc 15/16
Epoch 27 loss 0.25716587981037486 valid acc 15/16
Epoch 27 loss 0.4981065751787913 valid acc 15/16
Epoch 27 loss 0.3087551262704224 valid acc 15/16
Epoch 27 loss 0.24944110333495229 valid acc 15/16
Epoch 27 loss 0.31639205004879134 valid acc 15/16
Epoch 27 loss 0.16668182630273592 valid acc 15/16
Epoch 27 loss 0.35132433085904724 valid acc 15/16
Epoch 27 loss 0.21176928118175017 valid acc 15/16
Epoch 27 loss 0.1954704937402138 valid acc 15/16
Epoch 27 loss 0.26051274875680325 valid acc 15/16
Epoch 27 loss 0.06264423610030995 valid acc 15/16
Epoch 27 loss 0.28442626197159926 valid acc 16/16
Epoch 27 loss 0.04780052378099148 valid acc 16/16
Epoch 27 loss 0.6232880312977372 valid acc 15/16
Epoch 27 loss 0.1738665685032384 valid acc 15/16
Epoch 27 loss 0.13996351906560806 valid acc 15/16
Epoch 27 loss 0.0697957357784944 valid acc 14/16
Epoch 27 loss 0.13046116141415948 valid acc 15/16
Epoch 27 loss 0.21181446679334898 valid acc 15/16
Epoch 27 loss 0.03968776594610285 valid acc 15/16
Epoch 27 loss 0.07843451191077522 valid acc 15/16
Epoch 27 loss 0.15718016328402995 valid acc 15/16
Epoch 27 loss 0.21067419828865508 valid acc 15/16
Epoch 27 loss 0.1272605343125049 valid acc 15/16
Epoch 27 loss 0.3618760065724812 valid acc 15/16
Epoch 27 loss 0.38011818834935335 valid acc 15/16
Epoch 27 loss 0.10715740134701812 valid acc 15/16
Epoch 27 loss 0.7644829027683251 valid acc 15/16
Epoch 27 loss 0.08921193916554593 valid acc 15/16
Epoch 27 loss 0.1095694731296587 valid acc 15/16
Epoch 27 loss 0.3413301815374947 valid acc 15/16
Epoch 27 loss 0.21385553265084783 valid acc 16/16
Epoch 27 loss 0.16498262987611617 valid acc 16/16
Epoch 27 loss 0.16157387214779795 valid acc 16/16
Epoch 27 loss 0.4774287148070391 valid acc 16/16
Epoch 27 loss 0.3278449803070306 valid acc 15/16
Epoch 27 loss 0.1982852868049073 valid acc 15/16
Epoch 27 loss 0.2347435788295429 valid acc 15/16
Epoch 27 loss 0.13311690012975935 valid acc 15/16
Epoch 27 loss 0.39655842034229266 valid acc 15/16
Epoch 27 loss 0.3883209748627723 valid acc 15/16
Epoch 27 loss 0.30293862962046864 valid acc 15/16
Epoch 27 loss 0.08328993826543635 valid acc 15/16
Epoch 27 loss 0.1957246513913054 valid acc 15/16
Epoch 27 loss 0.09521634283587327 valid acc 15/16
Epoch 27 loss 0.11031081512763416 valid acc 15/16
Epoch 27 loss 0.23963775608894283 valid acc 15/16
Epoch 27 loss 0.25815015826473364 valid acc 15/16
Epoch 27 loss 0.21220442432320785 valid acc 15/16
Epoch 27 loss 0.33881355065933166 valid acc 14/16
Epoch 27 loss 0.031707028682030236 valid acc 14/16
Epoch 27 loss 0.5286835288799696 valid acc 15/16
Epoch 27 loss 0.10923868664014513 valid acc 15/16
Epoch 27 loss 0.2331555783033532 valid acc 15/16
Epoch 27 loss 0.21653548099292974 valid acc 15/16
Epoch 27 loss 0.051468331583799376 valid acc 15/16
Epoch 28 loss 0.00043697782277252184 valid acc 15/16
Epoch 28 loss 0.25455711392463976 valid acc 16/16
Epoch 28 loss 0.5660718307272969 valid acc 16/16
Epoch 28 loss 0.1122413706126496 valid acc 15/16
Epoch 28 loss 0.27952340243625956 valid acc 15/16
Epoch 28 loss 0.49093207397866656 valid acc 15/16
Epoch 28 loss 0.2683382890001031 valid acc 15/16
Epoch 28 loss 0.29507890117204555 valid acc 15/16
Epoch 28 loss 0.11822218202515483 valid acc 15/16
Epoch 28 loss 0.08964251967829956 valid acc 15/16
Epoch 28 loss 0.21393419706573336 valid acc 15/16
Epoch 28 loss 0.059694693138905075 valid acc 15/16
Epoch 28 loss 0.3605184373462834 valid acc 15/16
Epoch 28 loss 0.34537184367322454 valid acc 14/16
Epoch 28 loss 0.41763934008498654 valid acc 14/16
Epoch 28 loss 0.20912947404498602 valid acc 14/16
Epoch 28 loss 0.43239539294873175 valid acc 14/16
Epoch 28 loss 0.19140754235954954 valid acc 14/16
Epoch 28 loss 0.40033619958272587 valid acc 14/16
Epoch 28 loss 0.12165084676770782 valid acc 15/16
Epoch 28 loss 0.5853704278222218 valid acc 14/16
Epoch 28 loss 0.14345274865148655 valid acc 15/16
Epoch 28 loss 0.2023926158854235 valid acc 14/16
Epoch 28 loss 0.11686215050170368 valid acc 15/16
Epoch 28 loss 0.1684988902210493 valid acc 15/16
Epoch 28 loss 0.2371262769762042 valid acc 15/16
Epoch 28 loss 0.18983608928320145 valid acc 15/16
Epoch 28 loss 0.24623459743686238 valid acc 15/16
Epoch 28 loss 0.24338820059270402 valid acc 15/16
Epoch 28 loss 0.18732221992040088 valid acc 15/16
Epoch 28 loss 0.26426091649243105 valid acc 14/16
Epoch 28 loss 0.43289453956449125 valid acc 14/16
Epoch 28 loss 0.04200765002011547 valid acc 14/16
Epoch 28 loss 0.31572383079836625 valid acc 15/16
Epoch 28 loss 0.5417783462925185 valid acc 15/16
Epoch 28 loss 0.36456586821198506 valid acc 15/16
Epoch 28 loss 0.2833133327809855 valid acc 16/16
Epoch 28 loss 0.20234054976459853 valid acc 16/16
Epoch 28 loss 0.27096190692853195 valid acc 16/16
Epoch 28 loss 0.7034177317901336 valid acc 16/16
Epoch 28 loss 0.21669907310573555 valid acc 16/16
Epoch 28 loss 0.5526312313715583 valid acc 16/16
Epoch 28 loss 0.3090138053134603 valid acc 15/16
Epoch 28 loss 0.29120412204259677 valid acc 15/16
Epoch 28 loss 0.4684513428158145 valid acc 15/16
Epoch 28 loss 0.08724615733314706 valid acc 15/16
Epoch 28 loss 0.14420108027379536 valid acc 15/16
Epoch 28 loss 1.083368873410667 valid acc 15/16
Epoch 28 loss 0.31077262321319266 valid acc 15/16
Epoch 28 loss 0.11343349862063545 valid acc 15/16
Epoch 28 loss 0.8263678097841325 valid acc 15/16
Epoch 28 loss 0.08242407993546527 valid acc 15/16
Epoch 28 loss 0.5914804077047584 valid acc 15/16
Epoch 28 loss 0.1699225593785721 valid acc 15/16
Epoch 28 loss 0.31777189078171175 valid acc 15/16
Epoch 28 loss 0.24417478126277326 valid acc 15/16
Epoch 28 loss 0.353960444856779 valid acc 15/16
Epoch 28 loss 0.02115302441051481 valid acc 15/16
Epoch 28 loss 0.26618019303897167 valid acc 15/16
Epoch 28 loss 0.07420298266643444 valid acc 15/16
Epoch 28 loss 0.08069135381356562 valid acc 15/16
Epoch 28 loss 0.24832424003410447 valid acc 15/16
Epoch 28 loss 0.0565656392026494 valid acc 15/16
Epoch 29 loss 0.001888158448250743 valid acc 15/16
Epoch 29 loss 0.38918237711051357 valid acc 15/16
Epoch 29 loss 0.34544609097093054 valid acc 15/16
Epoch 29 loss 0.3684599443227496 valid acc 15/16
Epoch 29 loss 0.0377521902702343 valid acc 15/16
Epoch 29 loss 0.1169213615018802 valid acc 16/16
Epoch 29 loss 0.3467638247257757 valid acc 15/16
Epoch 29 loss 0.1369080423001524 valid acc 15/16
Epoch 29 loss 0.3035747196664559 valid acc 15/16
Epoch 29 loss 0.19767242874078167 valid acc 15/16
Epoch 29 loss 0.18466815009557513 valid acc 15/16
Epoch 29 loss 0.8866056777802052 valid acc 15/16
Epoch 29 loss 0.29935655553413865 valid acc 15/16
Epoch 29 loss 0.7623609084107625 valid acc 14/16
Epoch 29 loss 0.32862286019504783 valid acc 15/16
Epoch 29 loss 0.2930161130363355 valid acc 15/16
Epoch 29 loss 0.1540707259339368 valid acc 15/16
Epoch 29 loss 0.17139950074096977 valid acc 15/16
Epoch 29 loss 0.0663302931121933 valid acc 15/16
Epoch 29 loss 0.08866061640493367 valid acc 15/16
Epoch 29 loss 0.5055481891002762 valid acc 15/16
Epoch 29 loss 0.056542672848294506 valid acc 15/16
Epoch 29 loss 0.06031367868307297 valid acc 15/16
Epoch 29 loss 0.14307071130123633 valid acc 14/16
Epoch 29 loss 0.18911705672158777 valid acc 14/16
Epoch 29 loss 0.28022480592816884 valid acc 14/16
Epoch 29 loss 0.20244368047670613 valid acc 15/16
Epoch 29 loss 0.10625867690379259 valid acc 15/16
Epoch 29 loss 0.28452416990223384 valid acc 15/16
Epoch 29 loss 0.27005073808424745 valid acc 15/16
Epoch 29 loss 0.3073774358412251 valid acc 16/16
Epoch 29 loss 0.2621545547671933 valid acc 15/16
Epoch 29 loss 0.1770601689055574 valid acc 15/16
Epoch 29 loss 0.1948066055587009 valid acc 15/16
Epoch 29 loss 0.6901150529516009 valid acc 15/16
Epoch 29 loss 0.2914595899513942 valid acc 15/16
Epoch 29 loss 0.07097703156069024 valid acc 15/16
Epoch 29 loss 0.29540526201851214 valid acc 15/16
Epoch 29 loss 0.21041527745068211 valid acc 15/16
Epoch 29 loss 0.18302821624676557 valid acc 15/16
Epoch 29 loss 0.22887300746672734 valid acc 16/16
Epoch 29 loss 0.34440885148452427 valid acc 15/16
Epoch 29 loss 0.32656851829626565 valid acc 15/16
Epoch 29 loss 0.10589266350467044 valid acc 15/16
Epoch 29 loss 0.12696249464214415 valid acc 16/16
Epoch 29 loss 0.05713431638328714 valid acc 16/16
Epoch 29 loss 0.4155784846286512 valid acc 15/16
Epoch 29 loss 0.20583085901814746 valid acc 16/16
Epoch 29 loss 0.3434602185369412 valid acc 15/16
Epoch 29 loss 0.08639076771476062 valid acc 15/16
Epoch 29 loss 0.18577393625250305 valid acc 15/16
Epoch 29 loss 0.3115418029371741 valid acc 15/16
Epoch 29 loss 0.2926679453840198 valid acc 15/16
Epoch 29 loss 0.1411299421837931 valid acc 15/16
Epoch 29 loss 0.2771816271554315 valid acc 15/16
Epoch 29 loss 0.206718437504894 valid acc 15/16
Epoch 29 loss 0.24024769488053654 valid acc 15/16
Epoch 29 loss 0.11498213492884032 valid acc 15/16
Epoch 29 loss 0.5419064227669577 valid acc 15/16
Epoch 29 loss 0.649555492983656 valid acc 15/16
Epoch 29 loss 0.3149580421889919 valid acc 15/16
Epoch 29 loss 0.1479271214467488 valid acc 15/16
Epoch 29 loss 0.16420569645892386 valid acc 15/16
Epoch 30 loss 0.0007222626016775147 valid acc 15/16
Epoch 30 loss 0.11025790793141832 valid acc 15/16
Epoch 30 loss 0.12065647472948698 valid acc 15/16
Epoch 30 loss 0.368799470321561 valid acc 15/16
Epoch 30 loss 0.049900294694806946 valid acc 15/16
Epoch 30 loss 0.45432664725886823 valid acc 15/16
Epoch 30 loss 0.11133497645939255 valid acc 15/16
Epoch 30 loss 0.15727858920388404 valid acc 15/16
Epoch 30 loss 0.17426537954769705 valid acc 15/16
Epoch 30 loss 0.0853353793653216 valid acc 15/16
Epoch 30 loss 0.6048876108569424 valid acc 15/16
Epoch 30 loss 0.2691744891099952 valid acc 15/16
Epoch 30 loss 0.40843036003200917 valid acc 15/16
Epoch 30 loss 0.4670330614530309 valid acc 15/16
Epoch 30 loss 0.8093501673672091 valid acc 15/16
Epoch 30 loss 0.3751261699650594 valid acc 15/16
Epoch 30 loss 0.2886038288236624 valid acc 16/16
Epoch 30 loss 0.24280600944662656 valid acc 16/16
Epoch 30 loss 0.2599301906994759 valid acc 16/16
Epoch 30 loss 0.10887869823191926 valid acc 16/16
Epoch 30 loss 0.6000856685675457 valid acc 14/16
Epoch 30 loss 0.49256502264369817 valid acc 14/16
Epoch 30 loss 0.07037961648038904 valid acc 14/16
Epoch 30 loss 0.14497430709351455 valid acc 14/16
Epoch 30 loss 0.08125695464406246 valid acc 14/16
Epoch 30 loss 0.193950874380932 valid acc 15/16
Epoch 30 loss 0.04083188593870701 valid acc 15/16
Epoch 30 loss 0.12821835646184893 valid acc 15/16
Epoch 30 loss 0.2632809552601038 valid acc 15/16
Epoch 30 loss 0.020585703455427817 valid acc 15/16
Epoch 30 loss 0.21011482557303868 valid acc 15/16
Epoch 30 loss 0.23406732710523104 valid acc 15/16
Epoch 30 loss 0.18497965339206318 valid acc 15/16
Epoch 30 loss 0.0655675732106406 valid acc 16/16
Epoch 30 loss 0.7610393362757866 valid acc 14/16
Epoch 30 loss 0.18845468734464793 valid acc 15/16
Epoch 30 loss 0.0801772562162903 valid acc 15/16
Epoch 30 loss 0.09542641214652947 valid acc 15/16
Epoch 30 loss 0.0879991330784638 valid acc 15/16
Epoch 30 loss 0.09669209426227982 valid acc 15/16
Epoch 30 loss 0.04418487733723814 valid acc 15/16
Epoch 30 loss 0.30069395225141415 valid acc 15/16
Epoch 30 loss 0.16946640638659155 valid acc 15/16
Epoch 30 loss 0.07661700475436695 valid acc 15/16
Epoch 30 loss 0.4403089640987619 valid acc 16/16
Epoch 30 loss 0.05071633949687265 valid acc 16/16
Epoch 30 loss 0.17431974875975914 valid acc 15/16
Epoch 30 loss 0.4281801371482665 valid acc 15/16
Epoch 30 loss 0.17095972521972352 valid acc 15/16
Epoch 30 loss 0.035461126347227756 valid acc 15/16
Epoch 30 loss 0.05406209296196984 valid acc 15/16
Epoch 30 loss 0.495681893349564 valid acc 16/16
Epoch 30 loss 0.2718554651231375 valid acc 16/16
Epoch 30 loss 0.04905568573237368 valid acc 16/16
Epoch 30 loss 0.07339438656528635 valid acc 15/16
Epoch 30 loss 0.17600498434822884 valid acc 15/16
Epoch 30 loss 0.09388344662154485 valid acc 15/16
Epoch 30 loss 0.08453869438858963 valid acc 15/16
Epoch 30 loss 0.09132914405351256 valid acc 15/16
Epoch 30 loss 0.5143685138601775 valid acc 15/16
Epoch 30 loss 0.247418716193545 valid acc 15/16
Epoch 30 loss 0.01717078055912269 valid acc 15/16
Epoch 30 loss 0.06723402206785434 valid acc 16/16
real    13m22,620s
user    34m58,876s
sys     2m29,591s
```
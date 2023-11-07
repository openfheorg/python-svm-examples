from openfhe import *
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import roc_auc_score

# Load the data using pandas
print("---- Loading Data and Model ----")
X_test = pd.read_csv('data/credit_approval_train.csv')
x = X_test.to_numpy().flatten().tolist()
y_test = pd.read_csv('data/credit_approval_target_test.csv')
y_true = y_test.to_numpy().flatten().tolist()
y_pred_linear = np.loadtxt("data/y_pred_linear.txt")
# Get number of features
n = X_test.shape[1]
print("---- Data Loaded! ----")


# Load the model
weights = np.loadtxt("models/weights.txt")
intercept = np.loadtxt("models/intercept.txt")
print("---- Model Loaded! ----")

# Set up the CryptoContext
multDepth = 2
scaleModSize = 50
batchSize = n
parameters = CCParamsCKKSRNS()
parameters.SetMultiplicativeDepth(multDepth)
parameters.SetScalingModSize(scaleModSize)
parameters.SetBatchSize(batchSize)

cc = GenCryptoContext(parameters)

cc.Enable(PKE)
cc.Enable(KEYSWITCH)
cc.Enable(LEVELEDSHE)
cc.Enable(ADVANCEDSHE)
print(f"CKKS is using ring dimension {cc.GetRingDimension()}")

# Key Generation
keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)
cc.EvalSumKeyGen(keys.secretKey)

# Encoding and encryption of inputs
pt_x = cc.MakeCKKSPackedPlaintext(x)
pt_weights = cc.MakeCKKSPackedPlaintext(weights.tolist())
pt_bias = cc.MakeCKKSPackedPlaintext([intercept])

print(f"Input pt_weights: {pt_weights}")
print(f"Input pt_bias: {pt_bias}")

# Encrypt the encoded vectors
ct_x = cc.Encrypt(keys.publicKey, pt_x)

# Evaluation
t = time()
ct_res = cc.EvalInnerProduct(ct_x, pt_weights,n)
mask = [0] * n
mask[0] = 1
pt_mask = cc.MakeCKKSPackedPlaintext(mask)
ct_res = cc.EvalMult(ct_res, pt_mask)
ct_res = cc.EvalAdd(ct_res, pt_bias)
timeEvalSVMTime = time() - t
print(f"Linear-SVM inference took {timeEvalSVMTime} ms")

# Decryption and output

result = cc.Decrypt(ct_res, keys.secretKey)
# calculatee AUC betwwen y_true and result 
auc_enc = roc_auc_score(y_true, result)
auc_unenc = roc_auc_score(y_true, y_pred_linear)

print(f"Encrypted AUC: {auc_enc}")
print(f"Unencrypted AUC: {auc_unenc}")






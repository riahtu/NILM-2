import keras
net = keras.models.load_model('model.h5')
import zz_CNN as zz
import numpy as np


#prep data
# XTrain, YTrain, XTest, YTest = zz.prep_data('../data/self_made_data/P_data.csv',
#                                                     '../data/self_made_data/I_data.csv',
#                                                     '../data/self_made_data/S_data.csv',
#                                                     '../data/self_made_data/PF_data.csv',
#                                                     '../data/self_made_data/RP_data.csv',
#                                                     '../data/self_made_data/U_data.csv',
#                                                     '../data/self_made_data/target_data.csv', trainRate=1)
XTrain, YTrain, XTest, YTest = zz.prep_data('../data/self_made_data/P_data.csv',
                                                    '../data/self_made_data/I_data.csv',
                                                    '../data/self_made_data/S_data.csv',
                                                    '../data/self_made_data/PF_data.csv',
                                                    '../data/self_made_data/RP_data.csv',
                                                    '../data/self_made_data/U_data.csv',
                                                    '../data/self_made_data/target_data.csv', trainRate=1)

predictions=net.predict(XTrain)
zztrain=np.argmax(YTrain,axis=1)
zztarget=np.argmax(predictions,axis=1)
print(np.sum(zztrain == zztarget)/predictions[:,0].size)

from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from tensorflow.python.keras.engine import sequential
import My_Library                         
from My_Library import *

if 1==1:
    import ID3                         
    from ID3 import *

#Initial whole data at (this is for me): https://www.kaggle.com/c/deepfake-detection-challenge/discussion/128954
# https://www.kaggle.com/c/deepfake-detection-challenge/discussion/128954

# DATA/PATHS LOADING ________________________________________________________________________
if 1==1:
    # DeepFake ________________________________________________________________________
    if 1==2:
        filenames_train,  filenames_test = np.load('Dataset_DeepFake/filenames_train.npy', allow_pickle=True), np.load('Dataset_DeepFake/filenames_test.npy', allow_pickle=True)
        train_paths = create_paths ('Dataset_DeepFake/Train_Videos_Raw_JPG/',filenames_train) #
        test_paths = create_paths ('Dataset_DeepFake/Test_Videos_Raw_JPG/',filenames_test) #
        Labels_train,  Labels_test = np.load('Dataset_DeepFake/Labels_train.npy', allow_pickle=True), np.load('Dataset_DeepFake/Labels_test.npy', allow_pickle=True)
        Labels_train, Labels_test = decode_labels (Labels_train), decode_labels (Labels_test)
        # list(Labels_train).count(1) + list(Labels_test).count(1)


        # Initial whole data at (this is for me): https://www.kaggle.com/c/deepfake-detection-challenge/discussion/128954

        train_paths,Labels_train = Balance_Data (train_paths,Labels_train)
        from sklearn.preprocessing import OneHotEncoder
        onehot_encoder = OneHotEncoder()
        Labels_train_oh = onehot_encoder.fit_transform(Labels_train.reshape(-1,1)).toarray()#.astype(int)
        Labels_test_oh = onehot_encoder.fit_transform(Labels_test.reshape(-1,1)).toarray()#.astype(int)

        ###from random import shuffle
        ###ind_list = [_r_ for _r_ in range(len(train_paths))]
        ###shuffle(ind_list)
        # ind_list = np.load('ind_list.npy')
        # train_paths  = train_paths[ind_list]
        # Labels_train_oh  = Labels_train_oh[ind_list]
        # Labels_train  = Labels_train[ind_list]
        if 1==2:
            from random import shuffle
            ind_list = [_r_ for _r_ in range(len(train_paths))]
            shuffle(ind_list)
            train_paths  = train_paths[ind_list]
            Labels_train_oh  = Labels_train_oh[ind_list]
            Labels_train  = Labels_train[ind_list]

        print (len(Labels_train))
        print (len(Labels_test))


    # Pneymonia ________________________________________________________________________
    if 1==1:
        
        root_path = 'Dataset_Medical'

        CT_VP_tr_path = [
            os.path.join(os.getcwd(), root_path+"/Train/CT_VP_path234", x)
            for x in os.listdir(root_path+"/Train/CT_VP_path234")
        ] 
        CT_N_tr_path = [
            os.path.join(os.getcwd(), root_path+"/Train/CT_N_path", x)
            for x in os.listdir(root_path+"/Train/CT_N_path")
        ] 

        CT_VP_ts_path = [
            os.path.join(os.getcwd(), root_path+"/Test/CT_VP_path234", x)
            for x in os.listdir(root_path+"/Test/CT_VP_path234")
        ] 
        CT_N_ts_path = [
            os.path.join(os.getcwd(), root_path+"/Test/CT_N_path", x)
            for x in os.listdir(root_path+"/Test/CT_N_path")
        ] 

        VP_tr_labels = np.array([1 for _ in range(len(CT_VP_tr_path))])
        N_tr_labels = np.array([0 for _ in range(len(CT_N_tr_path))])
        VP_ts_labels = np.array([1 for _ in range(len(CT_VP_ts_path))])
        N_ts_labels = np.array([0 for _ in range(len(CT_N_ts_path))])


        train_paths = np.concatenate((pd.DataFrame(CT_VP_tr_path), pd.DataFrame(CT_N_tr_path)), axis=0)
        test_paths = np.concatenate((pd.DataFrame(CT_VP_ts_path), pd.DataFrame(CT_N_ts_path)), axis=0)
        Labels_train = np.concatenate((VP_tr_labels, N_tr_labels), axis=0)
        Labels_test = np.concatenate((VP_ts_labels, N_ts_labels), axis=0)
        # list(Labels_train).count(1) + list(Labels_test).count(1)
        s = 1


        # cnt1, cnt0 = 0,0
        # for l in Labels_train:
        #     if l == 1:
        #         cnt1+=1
        #     else:
        #         cnt0+=1

        # cnt_tr = cnt0+cnt1
        # for l in Labels_test:
        #     if l == 1:
        #         cnt1+=1
        #     else:
        #         cnt0+=1

        # cnt_total = cnt_tr + cnt0+cnt1


        from sklearn.preprocessing import OneHotEncoder
        onehot_encoder = OneHotEncoder()
        Labels_train_oh = onehot_encoder.fit_transform(Labels_train.reshape(-1,1)).toarray()#.astype(int)
        Labels_test_oh = onehot_encoder.fit_transform(Labels_test.reshape(-1,1)).toarray()#.astype(int)


        # from random import shuffle
        # ind_list = [_r_ for _r_ in range(len(train_paths))]
        # shuffle(ind_list)
        # train_paths  = train_paths[ind_list]
        # Labels_train_oh  = Labels_train_oh[ind_list]
        # Labels_train  = Labels_train[ind_list]




# _______ ΤXM (HC2)  _________________________________________________________________________________________
if 1==1:
    # HC Feature Extraction
    if 1==1:
        # Trial
        if 1==2:
            divider = 1
            width,heigth = 160,160
            n_channels = 3
            train_features = HC2_Feature_Extractor (divider,width,heigth,n_channels,np.array(['SAMPLE/aagfhgtpmv']).reshape(-1,1)) #HC1_Feature_Extractor (divider,width,heigth,n_channels,train_paths)#


        if 1==1:
            divider = 2
            width,heigth = 50,50
            n_channels = 3
            train_features = HC2_Feature_Extractor (divider,width,heigth,n_channels,train_paths) #HC1_Feature_Extractor (divider,width,heigth,n_channels,train_paths)#
            test_features = HC2_Feature_Extractor (divider,width,heigth,n_channels,test_paths)   #HC1_Feature_Extractor (divider,width,heigth,n_channels,test_paths)#
            np.save('Medical_Features/HC_40_50_train_features.npy',train_features)
            np.save('Medical_Features/HC_40_50_test_features.npy',test_features)
            s = 1



    # Evaluation
    if 1==1:# 
        train_features = np.load('DeepFake_Features/HC_25_160_train_features.npy')
        test_features  = np.load('DeepFake_Features/HC_25_160_test_features.npy')

        #train_features = np.load('HC_300_160_train_features_other2.npy')
        #test_features  = np.load('HC_300_160_test_features_other2.npy')

        f_train_seq, f_test_seq = train_features, test_features
        f_train_m, f_test_m = Standarize(np.mean(f_train_seq,axis=1)), Standarize(np.mean(f_test_seq,axis=1))#np.mean(f_train_seq,axis=1), np.mean(f_test_seq,axis=1)
        f_train_max, f_test_max = Standarize(np.max(f_train_seq,axis=1)), Standarize(np.max(f_test_seq,axis=1))
        f_train_min, f_test_min = Standarize(np.min(f_train_seq,axis=1)), Standarize(np.min(f_test_seq,axis=1))
        f_train_std, f_test_std = Standarize(np.std(f_train_seq,axis=1)), Standarize(np.std(f_test_seq,axis=1))

        #f_train,  f_test = np.concatenate ((f_train_m, f_train_max, f_train_min, f_train_std), axis = 1), np.concatenate ((f_test_m, f_test_max, f_test_min, f_test_std), axis = 1)

        # DeepFake, C=1, max_iter=1000
        # 300_160 1000:1250, 150_160 950:1250, 100_160 970:1485, 50_160 900:1450, 25_160  800:1450
        # GM = Evaluation(LogisticRegression(C=1, max_iter=1000).fit(f_train_m, Labels_train).predict(f_test_m), Labels_test)


        ## Medical_
        # 80_300 500:1000 C=10, max_iter=1000 || 80_150 900:1250 10,1000 || 80_75 200:250 10,1000 || 80_50 610:1000 1,1000 || 40_50 600:1000 10,1000 
        f_train = f_train_m [:,900:1450]
        f_test =  f_test_m [:,900:1450]
        GM = Evaluation(LogisticRegression(C=1, max_iter=1000).fit(f_train, Labels_train).predict(f_test), Labels_test)
        #GM = Evaluation(DecisionTreeClassifier(max_depth=5).fit(f_train, Labels_train).predict(f_test), Labels_test)
        s = 1


        #________ Feature Selection Algorithm _________
        if 1==1:

            # ___ Proposed ___
            if 1==2:
                c, it = 1,1000
                COEF_real = LOOCV_Features_Attribution (0.001, f_train, Labels_train, c, it) 
                COEF_abs = np.abs(COEF_real)
                ids_max, gm_max, IDS_DF, GM_VAL_DF = LOOCV_Feature_Reduction (COEF_abs, f_train, Labels_train, c, it) #
                
                GM_FS = Evaluation(LogisticRegression(C=1, max_iter=1000).fit(f_train[:,ids_max], Labels_train).predict(f_test[:,ids_max]), Labels_test)
  
            if 1==2: # maybe this since it removes corelated at first
                c, it = 1,1000
                ids_max, gm_max, IDS_DF, GM_VAL_DF = LOOCV_PeCo_Feature_Reduction (f_train, Labels_train, c, it)
                s = 1
                f_train , f_test = f_train[:,ids_max], f_test[:,ids_max]
                GM_FS = Evaluation(LogisticRegression(C=1, max_iter=1000).fit(f_train, Labels_train).predict(f_test), Labels_test)


                COEF_real = LOOCV_Features_Attribution (0.001, f_train, Labels_train, c, it) 
                COEF_abs = np.abs(COEF_real)
                ids_max, gm_max, IDS_DF, GM_VAL_DF = LOOCV_Feature_Reduction (COEF_abs, f_train, Labels_train, c, it) #

                GM_FS = Evaluation(LogisticRegression(C=1, max_iter=1000).fit(f_train[:,ids_max], Labels_train).predict(f_test[:,ids_max]), Labels_test)

        s = 1



            # ___ Pearson Corelation based ___




# 00:0.6169752314428771
# 01:0.628092796621031
# 02:0.628092796621031
# 03:0.624957911040728
# 04:0.624957911040728
# 05:0.6071959063489054
# 06:0.623834491273772
# 07:0.6444096124338013
# 08:0.628092796621031
# 09:0.6154574548966637
# 10:0.6137835657779984
# 11:0.6154574548966637
# 12:0.6097321448307896
# 13:0.5999127009691808
# 14:0.6323224255327735
# 15:0.6137835657779984
# 16:0.6276459144608478
# 17:0.6236095644623236
# 18:0.6252571487457086
# 19:0.6252571487457086




        #f_Total = np.concatenate((f_train, f_test),axis=0)
        #Labels_Total  =  np.concatenate((Labels_train, Labels_test.reshape(-1,1)),axis=0)


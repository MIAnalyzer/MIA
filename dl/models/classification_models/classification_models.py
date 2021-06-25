
# coding: utf-8

# In[4]:

def selection_genetica(target,df,modelo,train_size):
    
    from deap import creator, base, tools, algorithms #GENETIC ALGORITHM LIBRARY - requirement: pip install deap
    import random
    from sklearn import metrics,linear_model,svm,lda,cross_validation
    import pandas as pd
    import re 
    import numpy as np
    import statsmodels.api as sm
    import string

    # VARIABLES

    output_var = target

    list_inputs = set(df.drop(output_var,1).columns)



    print "GENETIC ALGORITHM FOR FEATURE SELECTION:"

    #####
    #SETING UP THE GENETIC ALGORITHM and CALCULATING STARTING POOL (STARTING CANDIDATE POPULATION)
    #####
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(list_inputs))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    def evalOneMax(individual):
        return sum(individual),

    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)



    NPOPSIZE = 50 #RANDOM STARTING POOL SIZE
    population = toolbox.population(n=NPOPSIZE)



    #####
    #ASSESSING GINI ON THE STARTING POOL
    #####
    dic_gini={}
    for i in range(np.shape(population)[0]): 

        # TRASLATING DNA INTO LIST OF VARIABLES (1-81)
        var_model = []    
        for j in range(np.shape(population)[1]): 
            if (population[i])[j]==1:
                var_model.append(list(list_inputs)[j])

        # ASSESSING GINI INDEX FOR EACH INVIVIDUAL IN THE INITIAL POOL 
        df_train,df_test=cross_validation.train_test_split(df,train_size=train_size)     
        X_train=df_train[var_model]
        Y_train=df_train[output_var]
        X_test=df_test[var_model]
        Y_test=df_test[output_var]
        X_train=X_train.values
        X_test=X_test.values
        ######
        # CHANGE_HERE - START: YOU ARE VERY LIKELY USING A DIFFERENT TECHNIQUE BY NOW. SO CHANGE TO YOURS.
        #####   

        if modelo=="LogisticRegression":
            clf=linear_model.LogisticRegression()

        elif modelo==("SVC"):

            clf=svm.SVC(probability=True)



        elif modelo=="LDA":
            clf = lda.LDA()

        clf.fit(X_train,Y_train)
        Y_predict=np.transpose(clf.predict_proba(X_test))[1]

        ######
        # CHANGE_HERE - END: YOU ARE VERY LIKELY USING A DIFFERENT TECHNIQUE BY NOW. SO CHANGE TO YOURS.
        #####             


        ######
        # CHANGE_HERE - START: HERE IT USES THE DEVELOPMENT GINI TO SELECT VARIABLES, YOU SHOULD A DIFFERENT GINI. EITHER THE OOT GINI OR THE SQRT(DEV_GINI*OOT_GINI)
        #####                
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_predict)
        auc = metrics.auc(fpr, tpr)
        gini_power = abs(2*auc-1)
        ######
        # CHANGE_HERE - END: HERE IT USES THE DEVELOPMENT GINI TO SELECT VARIABLES, YOU SHOULD A DIFFERENT GINI. EITHER THE OOT GINI OR THE SQRT(DEV_GINI*OOT_GINI)
        #####                

        gini=str(gini_power)+";"+str(population[i]).replace('[','').replace(', ','').replace(']','')
        dic_gini[gini]=population[i]   
    list_gini=sorted(dic_gini.keys(),reverse=True)



    #####
    #GENETIC ALGORITHM MAIN LOOP - START
    # - ITERATING MANY TIMES UNTIL NO IMPROVMENT HAPPENS IN ORDER TO FIND THE OPTIMAL SET OF CHARACTERISTICS (VARIABLES)
    #####
    sum_current_gini=0.0
    sum_current_gini_1=0.0
    sum_current_gini_2=0.0
    first=0    
    OK = 1
    a=0

    while OK:  #REPEAT UNTIL IT DO NOT IMPROVE, AT LEAST A LITLE, THE GINI IN 2 GENERATIONS
        a=a+1
        print 'loop ', a
        OK=0
        ####
        # GENERATING OFFSPRING - START
        ####

        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1) #CROSS-X PROBABILITY = 50%, MUTATION PROBABILITY=10%

        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population =toolbox.select(offspring, k=len(population))
        ####
        # GENERATING OFFSPRING - END
        ####

        sum_current_gini_2=sum_current_gini_1
        sum_current_gini_1=sum_current_gini
        sum_current_gini=0.0

        #####
        #ASSESSING GINI ON THE OFFSPRING - START
        #####
        for j in range(np.shape(population)[0]): 
            if population[j] not in dic_gini.values(): 
                var_model = [] 
                for i in range(np.shape(population)[1]): 
                    if (population[j])[i]==1:
                        var_model.append(list(list_inputs)[i])

                # CREACCION CONJUNTO TRAIN Y TEST

                df_train,df_test=cross_validation.train_test_split(df,train_size=train_size)     
                X_train=df_train[var_model]
                Y_train=df_train[output_var]
                X_test=df_test[var_model]
                Y_test=df_test[output_var]
                X_train=X_train.values
                X_test=X_test.values

                ######
                # CHANGE_HERE - START: YOU ARE VERY LIKELY USING A DIFFERENT TECHNIQUE BY NOW. SO CHANGE TO YOURS.
                #####   

                # diferentes modelos según la opción elegida

                if modelo=="LogisticRegression":
                    clf=linear_model.LogisticRegression()

                elif modelo==("SVC"):

                    clf=svm.SVC(probability=True)


                elif modelo=="LDA":
                    clf = lda.LDA()


                clf.fit(X_train,Y_train)
                Y_predict=np.transpose(clf.predict_proba(X_test))[1]
                ######
                # CHANGE_HERE - END: YOU ARE VERY LIKELY USING A DIFFERENT TECHNIQUE BY NOW. SO CHANGE TO YOURS.
                #####            


                ######
                # CHANGE_HERE - START: HERE IT USES THE DEVELOPMENT GINI TO SELECT VARIABLES, YOU SHOULD A DIFFERENT GINI. EITHER THE OOT GINI OR THE SQRT(DEV_GINI*OOT_GINI)
                #####                       
                fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_predict)
                auc = metrics.auc(fpr, tpr)
                gini_power = abs(2*auc-1)
                ######
                # CHANGE_HERE - END: HERE IT USES THE DEVELOPMENT GINI TO SELECT VARIABLES, YOU SHOULD A DIFFERENT GINI. EITHER THE OOT GINI OR THE SQRT(DEV_GINI*OOT_GINI)
                #####                       

                gini=str(gini_power)+";"+str(population[j]).replace('[','').replace(', ','').replace(']','')
                dic_gini[gini]=population[j]  
        #####
        #ASSESSING GINI ON THE OFFSPRING - END
        #####

        #####
        #SELECTING THE BEST FITTED AMONG ALL EVER CREATED POPULATION AND CURRENT OFFSPRING - START
        #####           
        list_gini=sorted(dic_gini.keys(),reverse=True)
        population=[]
        for i in list_gini[:NPOPSIZE]:
            population.append(dic_gini[i])
            gini=float(i.split(';')[0])
            sum_current_gini+=gini


        #####
        #SELECTING THE BEST FITTED AMONG ALL EVER CREATED POPULATION AND CURRENT OFFSPRING - END
        #####           

        #HAS IT IMPROVED AT LEAST A LITLE THE GINI IN THE LAST 2 GENERATIONS
        print 'sum_current_gini=', sum_current_gini, 'sum_current_gini_1=', sum_current_gini_1, 'sum_current_gini_2=', sum_current_gini_2
        if(sum_current_gini>sum_current_gini_1+0.0001 or sum_current_gini>sum_current_gini_2+0.0001):
            OK=1
    #####
    #GENETIC ALGORITHM MAIN LOOP - END
    #####


    gini_max=list_gini[0]        
    gini=float(gini_max.split(';')[0])
    features=gini_max.split(';')[1]


    ####
    # PRINTING OUT THE LIST OF FEATURES
    #####
    f=0
    for i in range(len(features)):
        if features[i]=='1':
            f+=1
            print 'feature ', f, ':', list(list_inputs)[i]
    print 'gini: ', gini

    lista_var=[]
    f=0
    for i in range(len(features)):
        if features[i]=='1':
            f+=1
            lista_var.append(list(list_inputs)[i])

    return lista_var


# In[5]:

# Aplicado a dataframe

def modelos(target,data,train_size):
    
    from sklearn import metrics
    from sklearn import linear_model,svm,cross_validation,tree,lda,ensemble
    import pandas as pd
    import numpy as np
    
    
    seed=284772184
    lista_modelos=["LogisticRegression","SVC","LDA","DecisionTreeClassifier","RandomForestClassifier"]
    lista_auc=[]
    profundidades=[1,2,3,4]
    
    for modelo in lista_modelos:
        print "CALCULO DE AUC PARA EL MODELO:",modelo
        lista_auc_tree=[]
        
        if modelo in ("LogisticRegression","LDA","SVC") :
            
            if modelo=="LogisticRegression":
                clf=linear_model.LogisticRegression()
            
            elif modelo==("SVC"):

                clf=svm.SVC(probability=True)


            else:
                clf=lda.LDA()

            
            lista_var=selection_genetica(target,data,modelo,train_size)
            
            lista_var.append(target)
            data2=data[lista_var]
            
            data_train,data_test=cross_validation.train_test_split(data2,train_size=train_size,random_state=seed)
        
            Y_train=data_train[target]
            X_train=data_train.drop(target,1)

            Y_test=data_test[target]
            X_test=data_test.drop(target,1)
             

            
            clf.fit(X_train,Y_train)

            Y_predict=np.transpose(clf.predict_proba(X_test))[1]
            
            fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_predict)
            auc = metrics.auc(fpr, tpr)
            lista_auc.append(auc)
        
        elif modelo in ("DecisionTreeClassifier","RandomForestClassifier"):
            
            if modelo=="DecisionTreeClassifier":
                clf = tree.DecisionTreeClassifier()
            
            else:
                clf = ensemble.RandomForestClassifier()
            
            
            data_train,data_test=cross_validation.train_test_split(data,train_size=train_size,random_state=seed)
            
            Y_train=data_train[target]
            X_train=data_train.drop(target,1)

            Y_test=data_test[target]
            X_test=data_test.drop(target,1)
            
            
            for d in profundidades:

                # ENTRENAMIENTO
                
                clf.set_params(max_depth=d)
                clf.fit(X_train,Y_train)


                # PREDICTION EN EL TEST
                Y_predict=np.transpose(clf.predict_proba(X_test))[1]
                fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_predict)
                auc = metrics.auc(fpr, tpr)
                
                lista_auc_tree.append(auc)
                
            lista_auc.append(np.min(lista_auc_tree))
                


    max1_pos=np.argmax(lista_auc)
    max1=np.max(lista_auc)
    lista_auc[max1_pos]=-2732
    max2_pos=np.argmax(lista_auc)
    max2=np.max(lista_auc)
    lista_auc[max2_pos]=-2732
    max3_pos=np.argmax(lista_auc)
    max3=np.max(lista_auc)
    lista_auc[max3_pos]=-2732
    max4_pos=np.argmax(lista_auc)
    max4=np.max(lista_auc)
    lista_auc[max4_pos]=-2732
    max5_pos=np.argmax(lista_auc)
    max5=np.max(lista_auc)
    

    print "\n"
    print "\n"
    print "Primero modelo con mayor auc:",lista_modelos[max1_pos],"con auc de:",max1
    print "Segundo modelo con mayor auc:",lista_modelos[max2_pos],"con auc de:",max2
    print "Tercero modelo con mayor auc:",lista_modelos[max3_pos],"con auc de:",max3
    print "Cuarto modelo con mayor auc:",lista_modelos[max4_pos],"con auc de:",max4
    print "Quinto modelo con mayor auc:",lista_modelos[max5_pos],"con auc de:",max5
            


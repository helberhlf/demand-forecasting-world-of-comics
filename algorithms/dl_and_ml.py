#-------------------------------------------------------
# @author Helber
#-------------------------------------------------------

# Importing library for manipulation and exploration of datasets.
import numpy as np
import pandas as pd

# Importing libraries needed for predictive modeling, Machine Learning and Deep Learning algorithms..
import xgboost as xgb
from xgboost import plot_tree
#-------------------------------------------------------

# Creating a function to select the best features
def feature_imp(features, target,param_imp,n_best_features):
    # Define o classificador, ou seja, instância um objeto da classe XGBRegressor
    reg_XBGR = xgb.XGBRFRegressor(verbosity=0, silent=True)

    # ajuste os dados
    reg_XBGR.fit(features, target)

    # selecionando os melhores parâmetros com grid search, que indicar a importância relativa de cada atributo para fazer previsões precisas:
    reg_XBGR_feature_imp = reg_XBGR.get_booster().get_score(importance_type=param_imp)

    # obtém nome das colunas
    keys = list(reg_XBGR_feature_imp.keys())

    # obtém scores das features
    values = list(reg_XBGR_feature_imp.values())

    # crianndo dataframe  com  k recusros principais
    xbg_best_features = pd.DataFrame(data=values, index=keys, columns=["score_XGBRFRegressor"]).sort_values(
        by="score_XGBRFRegressor", ascending=True).nlargest(n_best_features, columns="score_XGBRFRegressor")

    # Return the best features
    return xbg_best_features

# Creating a function to make things easier selecting model parameters XGB
def xgb_model_helper(X_train, y_train, PARAMETERS, V_PARAM_NAME=False, V_PARAM_VALUES=False, BR=10):
    # Cria uma matrix temporária em formato de bit do conjunto de dados a ser treinados
    temp_dmatrix = xgb.DMatrix(data=X_train, label=y_train)

    # Check os parâmetros a ser utilizados
    if V_PARAM_VALUES == False:
        cv_results = xgb.cv(dtrain=temp_dmatrix, nfold=5, num_boost_round=BR, params=PARAMETERS, as_pandas=True,
                            seed=123)
        return cv_results

    else:
        # Criando uma Lista, para armazenar os resultados e os nomes, de cada uma das métricas.
        results = []

        # Percorre a lista de parâmetros
        for v_param_value in V_PARAM_VALUES:
            # Adicionando o nome dos parâmetros avaliado a lista de nomes.
            PARAMETERS[V_PARAM_NAME] = v_param_value

            # Treinando o modelo com Cross Validation.
            cv_results = xgb.cv(dtrain=temp_dmatrix, nfold=5, num_boost_round=BR, params=PARAMETERS, as_pandas=True,
                                seed=123)

            # Adicionando os resultados gerados a lista de resultados.
            results.append((cv_results["train-mae-mean"].tail().values[-1], cv_results["test-mae-mean"].tail().values[
                -1]))  # .tail().values[-1] captura somente as colunas

        # zip “pareia” os elementos de uma série de listas, tuplas ou outras sequências para criar uma lista de tuplas:

        # Adicionando a média da AUC e o desvio-padrão dos resultados gerados, pelo modelo analisado ao Dataframe de médias.
        data = list(zip(V_PARAM_VALUES, results))
        print(pd.DataFrame(data, columns=[V_PARAM_NAME, "mae"]))

        return cv_results
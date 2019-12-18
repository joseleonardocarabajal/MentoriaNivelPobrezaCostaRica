# Importación de las librerías necesarias
import numpy as np
import pandas as pd
np.random.seed(0)

def exec_script(_ds, _data_fields):
    _ds_reduced = _ds
    _data_fields_reduced = _data_fields

    #Features Tipo Objetos    
    _ds_reduced['dependencycalculated'] = _ds_reduced[['hogar_nin','hogar_mayor']].sum(axis=1).div(_ds_reduced['hogar_adul'])
    _ds_reduced['dependencycalculated']=_ds_reduced['dependencycalculated'].replace(np.inf, -1)
    _ds_reduced.drop(columns = 'dependency', inplace = True)
    _ds_reduced.loc[_ds_reduced.edjefe =='yes', 'edjefe'] = 1
    _ds_reduced.loc[_ds_reduced.edjefa =='yes', 'edjefa'] = 1
    _ds_reduced.loc[(_ds_reduced.edjefe =='no') & (_ds_reduced.male == 1), 'edjefe'] = 0
    _ds_reduced.loc[(_ds_reduced.edjefa =='no') & (_ds_reduced.female == 1), 'edjefa'] = 0
    _ds_reduced.loc[(_ds_reduced.edjefe =='no') & (_ds_reduced.male == 0), 'edjefe'] = float('nan')
    _ds_reduced.loc[(_ds_reduced.edjefa =='no') & (_ds_reduced.female == 0), 'edjefa'] = float('nan')
    _ds_reduced.loc[_ds_reduced.male == 0, 'edjefe'] = _ds_reduced['edjefa']
    _ds_reduced.loc[_ds_reduced.female == 0, 'edjefa'] = _ds_reduced['edjefe']
    _ds_reduced['edjefe'] = _ds_reduced['edjefe'].astype(np.int64)
    _ds_reduced['edjefa'] = _ds_reduced['edjefa'].astype(np.int64)
    _ds_reduced['edjef']=_ds_reduced['edjefe']
    _ds_reduced.drop(columns = 'edjefe', inplace = True)
    _ds_reduced.drop(columns = 'edjefa', inplace = True)

    _data_fields_reduced['Variable_name'].replace(['dependency'], 'dependencycalculated',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['edjefa'], 'edjef',inplace=True)
    _data_fields_reduced = _data_fields_reduced[(_data_fields_reduced.Variable_name != 'edjefe')]

    #Campos tamhog y hhsize
    _ds_reduced.drop(columns = 'tamhog', inplace = True)
    _ds_reduced.drop(columns = 'hogar_total', inplace = True)

    _data_fields_reduced = _data_fields_reduced[(_data_fields_reduced.Variable_name != 'tamhog') & (_data_fields_reduced.Variable_name != 'hogar_total')]

    #Renta Mensual y Tipo de Vivienda
    meat_target_1 = _ds_reduced[(_ds_reduced['v2a1']>0) & (_ds_reduced['v2a1']!='') & (_ds_reduced['v2a1'].notnull()) & (_ds_reduced['tipovivi2']==1) & (_ds_reduced['Target']==1)]['v2a1'].mean()
    meat_target_2 = _ds_reduced[(_ds_reduced['v2a1']>0) & (_ds_reduced['v2a1']!='') & (_ds_reduced['v2a1'].notnull()) & (_ds_reduced['tipovivi2']==1) & (_ds_reduced['Target']==2)]['v2a1'].mean()
    meat_target_3 = _ds_reduced[(_ds_reduced['v2a1']>0) & (_ds_reduced['v2a1']!='') & (_ds_reduced['v2a1'].notnull()) & (_ds_reduced['tipovivi2']==1) & (_ds_reduced['Target']==3)]['v2a1'].mean()
    meat_target_4 = _ds_reduced[(_ds_reduced['v2a1']>0) & (_ds_reduced['v2a1']!='') & (_ds_reduced['v2a1'].notnull()) & (_ds_reduced['tipovivi2']==1) & (_ds_reduced['Target']==4)]['v2a1'].mean()
    _ds_reduced.loc[(_ds_reduced['v2a1']==0) | (_ds_reduced['v2a1']=='') | (_ds_reduced['v2a1'].isnull()) & (_ds_reduced['tipovivi2']==1) & (_ds_reduced['Target']==1), 'v2a1'] = meat_target_1
    _ds_reduced.loc[(_ds_reduced['v2a1']==0) | (_ds_reduced['v2a1']=='') | (_ds_reduced['v2a1'].isnull()) & (_ds_reduced['tipovivi2']==1) & (_ds_reduced['Target']==2), 'v2a1'] = meat_target_2
    _ds_reduced.loc[(_ds_reduced['v2a1']==0) | (_ds_reduced['v2a1']=='') | (_ds_reduced['v2a1'].isnull()) & (_ds_reduced['tipovivi2']==1) & (_ds_reduced['Target']==3), 'v2a1'] = meat_target_3
    _ds_reduced.loc[(_ds_reduced['v2a1']==0) | (_ds_reduced['v2a1']=='') | (_ds_reduced['v2a1'].isnull()) & (_ds_reduced['tipovivi2']==1) & (_ds_reduced['Target']==4), 'v2a1'] = meat_target_4

    #Renta Mensual: Valores Faltantes
    _ds_reduced.loc[(_ds_reduced['v2a1']=='') | (_ds_reduced['v2a1'].isnull()), 'v2a1'] = 0

    #Tratamiento de Valores Faltantes   
    _ds_reduced['v2a1'] = _ds_reduced['v2a1'].fillna(0)
    _ds_reduced['v18q1'] = _ds_reduced['v18q1'].fillna(0)
    _ds_reduced['rez_esc'] = _ds_reduced['rez_esc'].fillna(0)
    _ds_reduced['meaneduc'] = _ds_reduced['meaneduc'].fillna(0)

    #Otras Nuevas columnas
    _ds_reduced['r4h3_r4t3'] = _ds_reduced['r4h3'].div(_ds_reduced['r4t3'])
    _ds_reduced['r4m3_r4t3'] = _ds_reduced['r4m3'].div(_ds_reduced['r4t3'])
    _ds_reduced['hogar_nin_r4t3'] = _ds_reduced['hogar_nin'].div(_ds_reduced['r4t3'])
    _ds_reduced['hogar_adul_r4t3'] = _ds_reduced['hogar_adul'].div(_ds_reduced['r4t3'])
    _ds_reduced['hogar_mayor_r4t3'] = _ds_reduced['hogar_mayor'].div(_ds_reduced['r4t3'])

    _data_fields_reduced.append({'Variable_name':'r4h3_r4t3', 'Variable_description':'Reason Male/Total persons'}, ignore_index=True)
    _data_fields_reduced.append({'Variable_name':'r4m3_r4t3', 'Variable_description':'Reason Female/Total persons'}, ignore_index=True)
    _data_fields_reduced.append({'Variable_name':'hogar_nin_r4t3', 'Variable_description':'Reason Children/Total persons'}, ignore_index=True)
    _data_fields_reduced.append({'Variable_name':'hogar_adul_r4t3', 'Variable_description':'Reason Adults/Total persons'}, ignore_index=True)
    _data_fields_reduced.append({'Variable_name':'hogar_mayor_r4t3', 'Variable_description':'Reason 65+/Total persons'}, ignore_index=True)

    #Eliminar variables consideradas irrelevantes
    def get_categorical_cols(col_regx):
        return _ds_reduced.columns.str.extractall(r'^({})$'.format(col_regx))[0].values.tolist()
    cols = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe','SQBhogar_nin', 
            'SQBovercrowding','SQBdependency', 'SQBmeaned','agesq']
    dropped_cols = []
    for col in cols:
        dropped_cols += get_categorical_cols(col)
    _ds_reduced.drop(dropped_cols,axis=1)

    _data_fields_reduced = _data_fields_reduced[(_data_fields_reduced.Variable_name != 'SQBescolari') & (_data_fields_reduced.Variable_name != 'SQBage') & (_data_fields_reduced.Variable_name != 'SQBhogar_total') & (_data_fields_reduced.Variable_name != 'SQBedjefe') & (_data_fields_reduced.Variable_name != 'SQBhogar_nin') & (_data_fields_reduced.Variable_name != 'SQBovercrowding') & (_data_fields_reduced.Variable_name != 'SQBdependency') & (_data_fields_reduced.Variable_name != 'SQBmeaned') & (_data_fields_reduced.Variable_name != 'agesq')]

    #Renombran las variables
    _col_dict = {'v2a1': 'Renta',
                 'age': 'Edad',
                 'escolari': 'Anios_escolaridad_hechos',
                 'rez_esc': 'Anios_escolaridad_faltantes',
                 'meaneduc': 'Educ_media',
                 'hacdor': 'Exceso_habitaciones',
                 'rooms': 'Habitaciones',
                 'v14a': 'Tiene_banio',
                 'hhsize': 'Tamanio_hogar_hhsize',
                 'v18q': 'Tiene_tablet',
                 'v18q1': 'Cant_tablets',
                 'r4h1': 'Cant_hom_menores_12',
                 'r4h2': 'Cant_hom_mayores_12',
                 'r4h3': 'Total_hombres',
                 'r4m1': 'Cant_muj_menores_12',
                 'r4m2': 'Cant_muj_mayores_12',
                 'r4m3': 'Total_mujeres',
                 'r4t1': 'Cant_indiv_menores_12',
                 'r4t2': 'Cant_indiv_mayores_12',
                 'r4t3': 'Total_indiv',
                 'tamviv': 'Tamanio_vivienda',
                 'dependencycalculated': 'Cant_habitantes_depend',
                 'edjef': 'Educ_jefe',
                 'hacapo': 'Exceso_ambientes',
                 'public': 'Electridad_publica',
                 'planpri': 'Electridad_privada',
                 'noelec': 'Electridad_no_tiene',
                 'coopele': 'Electridad_cooperativa',
                 'overcrowding': 'Indiv_x_ambientes',
                 'computer': 'Tiene_pc',
                 'television': 'Tiene_tv',
                 'mobilephone': 'Tiene_cel',
                 'qmobilephone': 'Cant_cel',
                 'lugar1': 'Central',
                 'lugar2': 'Chorotega',
                 'lugar3': 'Pacifico_Central',
                 'lugar4': 'Brunca',
                 'lugar5': 'Huetar_Atlantica',
                 'lugar6': 'Huetar_Norte',
                 'r4h3_r4t3': 'Razon_Hombres_Total',
                 'r4m3_r4t3': 'Razon_Mujeres_Total',
                 'hogar_nin_r4t3': 'Razon_Ninios_Total',
                 'hogar_adul_r4t3': 'Razon_Adultos_Total',
                 'hogar_mayor_r4t3': 'Razon_Ancianos_Total'
                }

    _ds_reduced = _ds_reduced.rename(columns=_col_dict)

    _data_fields_reduced['Variable_name'].replace(['v2a1'], 'Renta',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['age'], 'Edad',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['escolari'], 'Anios_escolaridad_hechos',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['rez_esc'], 'Anios_escolaridad_faltantes',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['meaneduc'], 'Educ_media',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['hacdor'], 'Exceso_habitaciones',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['rooms'], 'Habitaciones',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['v14a'], 'Tiene_banio',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['hhsize'], 'Tamanio_hogar_hhsize',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['v18q'], 'Tiene_tablet',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['v18q1'], 'Cant_tablets',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['v2a1'], 'Renta',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['r4h1'], 'Cant_hom_menores_12',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['r4h2'], 'Cant_hom_mayores_12',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['r4h3'], 'Total_hombres',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['r4m1'], 'Cant_muj_menores_12',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['r4m2'], 'Cant_muj_mayores_12',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['r4m3'], 'Total_mujeres',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['r4t1'], 'Cant_indiv_menores_12',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['r4t2'], 'Cant_indiv_mayores_12',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['r4t3'], 'Total_indiv',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['tamviv'], 'Tamanio_vivienda',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['dependencycalculated'], 'Cant_habitantes_depend',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['edjef'], 'Educ_jefe',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['hacapo'], 'Exceso_ambientes',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['v14a'], 'Tiene_banio',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['public'], 'Electridad_publica',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['planpri'], 'Electridad_privada',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['noelec'], 'Electridad_no_tiene',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['coopele'], 'Electridad_cooperativa',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['overcrowding'], 'Indiv_x_ambientes',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['computer'], 'Tiene_pc',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['television'], 'Tiene_tv',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['mobilephone'], 'Tiene_cel',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['qmobilephone'], 'Cant_cel',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['lugar1'], 'Central',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['lugar2'], 'Chorotega',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['lugar3'], 'Pacifico_Central',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['lugar4'], 'Brunca',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['lugar5'], 'Huetar_Atlantica',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['lugar6'], 'Huetar_Norte',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['r4h3_r4t3'], 'Razon_Hombres_Total',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['Xr4m3_r4t3XX'], 'Razon_Mujeres_Total',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['hogar_nin_r4t3'], 'Razon_Ninios_Total',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['hogar_adul_r4t3'], 'Razon_Adultos_Total',inplace=True)
    _data_fields_reduced['Variable_name'].replace(['hogar_mayor_r4t3'], 'Razon_Ancianos_Total',inplace=True)

    #Reordenar variables
    ind_cols = ['Id','Edad','female', 'male','dis','estadocivil1', 'estadocivil2',
                'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6',
                'estadocivil7','parentesco1', 'parentesco2', 'parentesco3',
                'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7',
                'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11',
                'parentesco12','Anios_escolaridad_hechos','Anios_escolaridad_faltantes','instlevel1', 
                'instlevel2','instlevel3', 'instlevel4', 'instlevel5', 'instlevel6',
                'instlevel7', 'instlevel8', 'instlevel9',
                #'SQBescolari','SQBage', 'SQBhogar_total', 'SQBedjefe','SQBhogar_nin', 
                #'SQBovercrowding', 'SQBdependency', 'SQBmeaned','agesq'
               ]
    habitantes_cols = ['Tiene_tablet', 'Cant_tablets', 'Cant_hom_menores_12', 'Cant_hom_mayores_12', 'Total_hombres', 'Cant_muj_menores_12', 'Cant_muj_mayores_12', 'Total_mujeres',
                       'Cant_indiv_menores_12', 'Cant_indiv_mayores_12', 'Total_indiv','Tamanio_vivienda','Cant_habitantes_depend','Educ_jefe','Razon_Hombres_Total',
                      'Razon_Mujeres_Total','Razon_Ninios_Total','Razon_Adultos_Total','Razon_Ancianos_Total']
    hogar_cols = ['idhogar','Renta', 'hogar_nin', 'hogar_adul',
                   'hogar_mayor','Exceso_habitaciones', 'Habitaciones', 'Exceso_ambientes', 'Tiene_banio', 'refrig',
                   'Tamanio_hogar_hhsize','Educ_media','paredblolad', 'paredzocalo', 'paredpreb', 'pareddes',
                   'paredmad', 'paredzinc', 'paredfibras', 'paredother', 'pisomoscer',
                   'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene',
                   'pisomadera', 'techozinc', 'techoentrepiso', 'techocane',
                   'techootro', 'cielorazo', 'abastaguadentro', 'abastaguafuera',
                   'abastaguano','Electridad_publica', 'Electridad_privada', 'Electridad_no_tiene', 'Electridad_cooperativa',
                   'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5',
                   'sanitario6', 'energcocinar1', 'energcocinar2', 'energcocinar3',
                   'energcocinar4', 'elimbasu1', 'elimbasu2', 'elimbasu3',
                   'elimbasu4', 'elimbasu5', 'elimbasu6', 'epared1', 'epared2',
                   'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2',
                   'eviv3','bedrooms', 'Indiv_x_ambientes', 'tipovivi1', 'tipovivi2',
                   'tipovivi3', 'tipovivi4', 'tipovivi5', 'Tiene_pc', 'Tiene_tv',
                   'Tiene_cel', 'Cant_cel', 'Central', 'Chorotega', 'Pacifico_Central',
                   'Brunca', 'Huetar_Atlantica', 'Huetar_Norte', 'area1', 'area2','Target']
    _ds_reduced = _ds_reduced[ind_cols + habitantes_cols + hogar_cols]

    #Filtrar hogares que tienen más de un Target asignado
    _hogar_dup =_ds_reduced.groupby('idhogar')['Target'].nunique().reset_index().query('Target > 1')
    _ds_reduced = _ds_reduced[~_ds_reduced['idhogar'].isin(list(_hogar_dup['idhogar']))]

    #Reducción del dataset
    for col in _ds_reduced.columns:
        _ds_reduced[col]= pd.to_numeric(_ds_reduced[col],downcast='unsigned',errors='ignore')

    #Limpiar correlaciones (solo dummies)
    _ds_reduced.drop(columns = ['male','estadocivil7','parentesco12','instlevel9','paredother','pisonotiene',
                                'techootro','abastaguano','Electridad_no_tiene','sanitario6','energcocinar4',
                                'elimbasu6','epared3','etecho3','eviv3','tipovivi5','Huetar_Norte','area2'],
                     inplace = True)
    _data_fields_reduced = _data_fields_reduced[(_data_fields_reduced.Variable_name != 'male') & (_data_fields_reduced.Variable_name != 'estadocivil7') & (_data_fields_reduced.Variable_name != 'parentesco12') & (_data_fields_reduced.Variable_name != 'instlevel9') & (_data_fields_reduced.Variable_name != 'paredother') & (_data_fields_reduced.Variable_name != 'pisonotiene') & (_data_fields_reduced.Variable_name != 'techootro') & (_data_fields_reduced.Variable_name != 'abastaguano') & (_data_fields_reduced.Variable_name != 'Electridad_no_tiene') & (_data_fields_reduced.Variable_name != 'sanitario6') & (_data_fields_reduced.Variable_name != 'energcocinar4') & (_data_fields_reduced.Variable_name != 'elimbasu6') & (_data_fields_reduced.Variable_name != 'epared3') & (_data_fields_reduced.Variable_name != 'etecho3') & (_data_fields_reduced.Variable_name != 'eviv3') & (_data_fields_reduced.Variable_name != 'tipovivi5') & (_data_fields_reduced.Variable_name != 'Huetar_Norte') & (_data_fields_reduced.Variable_name != 'area2')]
    hogar_cols.remove('paredother')
    hogar_cols.remove('pisonotiene')
    hogar_cols.remove('techootro')
    hogar_cols.remove('abastaguano')
    hogar_cols.remove('Electridad_no_tiene')
    hogar_cols.remove('sanitario6')
    hogar_cols.remove('energcocinar4')
    hogar_cols.remove('elimbasu6')
    hogar_cols.remove('epared3')
    hogar_cols.remove('etecho3')
    hogar_cols.remove('eviv3')
    hogar_cols.remove('tipovivi5')
    hogar_cols.remove('Huetar_Norte')
    hogar_cols.remove('area2')
    
    return _ds_reduced, _data_fields_reduced
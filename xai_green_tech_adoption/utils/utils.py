# official keys of municipalities and municipality associations
col_id_m = 'Official municipality code (AGS)'
col_id_ma = 'Code of municipality associations (RS)'
col_id_c = 'County code'
col_id_s = 'State code'

col_name_m = 'Municipality name'
col_name_ma = 'Name of municipality ass.'
col_name_c = 'County name'

####
col_state_code = 'state'
col_nuts2_code = 'nuts2 region'
col_county_code = 'county'
col_ma_code = 'municipality association'
col_m_code = 'municipality'
###

map_state_rs = {1: 'Schleswig-Holstein',
                2: 'Hamburg',
                3: 'Lower Saxony',
                4: 'Bremen',
                5: 'North Rhine-Westphalia',
                6: 'Hesse',
                7: 'Rhineland-Palatinate',
                8: 'Baden-Württemberg',
                9: 'Bavaria',
                10: 'Saarland',
                11: 'Berlin',
                12: 'Brandenburg',
                13: 'Mecklenburg-Vorpommern',
                14: 'Saxony',
                15: 'Saxony-Anhalt',
                16: 'Thuringia'}

# INKAR

# base year of INKAR data set
t_inkar = 2019

# columns of INKAR data frame
col_name_inkar = 'Name'
col_name_c_inkar = 'Name Kreise'
col_spatial_inkar = 'Raumbezug'
col_time_inkar = 'Zeitbezug'
col_value_inkar = 'Wert'
col_id_general_inkar = 'Kennziffer'
col_ind_inkar = 'Indikator'
col_ars_ma_inkar = 'Regionalschlüssel'

relevant_statial_entities = ['Gemeinden', 'Gemeindeverbände', 'Kreise']
# Gemeinden = municipalities (m)
# Gemeindeverbände = municipality associations (ma)
# Kreise = counties (c)

# indicators taken from INKAR dataset

# 1. indicators available on ma-level
# 1a. available for 2019
add_ind_ma_2019 = [
    # Absolutzahlen
    'Arbeitslose',
    'sozialversicherungspflichtig Beschäftigte am Arbeitsort',
    'sozialversicherungspflichtig Beschäftigte am Wohnort',
    'Bevölkerung gesamt',
    'Bevölkerung männlich',
    'Bevölkerung weiblich',
    'Erwerbsfähige Bevölkerung (15 bis unter 65 Jahre)',
    'Bevölkerung (mit Korrektur VZ 1987/Zensus 2011)',
    'Bodenfläche gesamt',
    # Arbeitslosigkeit - allgemein
    # 'Arbeitslose Frauen',
    # 'Arbeitslose Männer',
    # Bauen und Wohnen - Baulandmarkt und Bautätigkeit
    'Baugenehmigungen für Wohnungen',
    'Baugenehmigungen für Wohnungen in Ein- und Zweifamilienhäusern',
    'Baugenehmigungen für Wohnungen in Mehrfamilienhäusern',
    'Fertiggestellte Wohnungen im Bestand',
    'Neue Ein- und Zweifamilienhäuser',
    'Neubauwohnungen in Ein- und Zweifamilienhäusern',
    'Neubauwohnungen je Einwohner',
    'Neubauwohnungen in Ein- und Zweifamilienhäusern je Einwohner',
    'Neubauwohnungen in Mehrfamilienhäusern',
    # Bauen und Wohnen - Gebäude- und Wohnungsbestand
    'Ein- und Zweifamilienhäuser',
    'Mehrfamilienhäuser',
    'Ein- und Zweiraumwohnungen',
    '5- und mehr Raum-Wohnungen',
    'Wohnungen in Ein- und Zweifamilienhäusern',
    'Wohnungen in Mehrfamilienhäusern',
    # Beschäftigung und Erwerbstätigkeit – Struktur
    'Beschäftigtenquote',
    'Beschäftigtenquote Frauen',
    'Beschäftigtenquote Männer',
    'Verhältnis junge zu alte Erwerbsfähige',
    # Bevölkerung – Altersstruktur
    'Einwohner unter 6 Jahre',
    'Einwohner von 6 bis unter 18 Jahren',
    'Einwohner von 18 bis unter 25 Jahren',
    'Einwohner von 25 bis unter 30 Jahren',
    'Einwohner von 30 bis unter 50 Jahren',
    'Einwohner von 50 bis unter 65 Jahren',
    'Einwohner 65 Jahre und älter',
    'Einwohner 75 Jahre und älter',
    'Weibliche Einwohner 75 Jahre und älter',
    'Weibliche Einwohner von 18 bis unter 25 Jahren',
    'Weibliche Einwohner von 25 bis unter 30 Jahren',
    'Weibliche Einwohner 65 Jahre und älter',
    'Einwohner von 65 bis unter 75 Jahren',
    'Weibliche Einwohner von 65 bis unter 75 Jahren',
    'Einwohner unter 3 Jahren',
    'Einwohner von 3 bis unter 6 Jahren',
    'Durchschnittsalter der Bevölkerung',
    # Bevölkerung – Bevölkerungsstruktur
    'Abhängigenquote Junge',
    'Abhängigenquote Alte',
    'Frauenanteil',
    'Frauenanteil 20 bis unter 40 Jahre',
    # Bevölkerung – Wanderungen
    'Gesamtwanderungssaldo',
    # Bevölkerung - Natürliche Bevölkerungsbewegungen
    'Geborene',
    'Gestorbene',
    'Natürlicher Saldo',
    # Flächennutzung
    'Siedlungs- und Verkehrsfläche',
    'Siedlungsdichte in km²',
    'Erholungsfläche',
    'Erholungsfläche je Einwohner',
    'Freifläche',
    'Freifläche je Einwohner',
    'Landwirtschaftsfläche',
    'naturnähere Fläche',
    'Naturnähere Fläche je Einwohner',
    'Waldfläche',
    'Wasserfläche',
    # Öffentliche Finanzen
    'Steuerkraft',
    'Einkommensteuer',
    'Gewerbesteuer',
    'Umsatzsteuer',
    # Siedlungsstruktur
    'Einwohnerdichte',
    'Einwohner-Arbeitsplatz-Dichte',
    'Regionales Bevölkerungspotenzial',
    # Verkehr und Erreichbarkeit - Pendler
    'Einpendler',
    'Auspendler',
    'Pendlersaldo',
    'Pendler mit Arbeitsweg 50 km und mehr',
    'Pendler mit Arbeitsweg 150 km und mehr',
    'Pendler mit Arbeitsweg 300 km und mehr',
]
# 1b. only available for other time periods: dictionary gives name of indicator and year considered
add_ind_ma_dict = {
    # Erreichbarkeit
    'Erreichbarkeit von Autobahnen': 2020,
    'Erreichbarkeit von Flughäfen': 2020,
    'Erreichbarkeit von IC/EC/ICE-Bahnhöfen': 2020,
    'Erreichbarkeit von Oberzentren': 2020,
    'Erreichbarkeit von Mittelzentren': 2020,
    'Nahversorgung Supermärkte Durchschnittsdistanz': 2017,
    'Nahversorgung Supermärkte Anteil der Bev. 1km Radius': 2017,
    'Nahversorgung Apotheken Durchschnittsdistanz': 2017,
    'Nahversorgung Apotheken Anteil der Bev. 1km Radius': 2017,
    'Nahversorgung Grundschulen Durchschnittsdistanz': 2018,
    'Nahversorgung Grundschulen Anteil der Bev. 1km Radius': 2018,
    'Nahversorgung Haltestellen des ÖV Durchschnittsdistanz': 2018,
    'Nahversorgung Haltestellen des ÖV Anteil der Bev. 1km Radius': 2018
}

# 2. indicators available on c-level
# 2a. available for 2019
add_ind_c_2019 = [
    # Arbeitslosigkeit - allgemein
    'Arbeitslosenquote',
    'Arbeitslosenquote Frauen',
    'Arbeitslosenquote Männer',
    # Bauen und Wohnen - Baulandmarkt und Bautätigkeit
    'Baulandpreise',
    'Fertiggestellte Wohngebäude mit erneuerbarer Heizenergie',
    'Fertiggestellte Wohnungen mit erneuerbarer Heizenergie',
    # Wohnfläche
    'Wohnfläche',
    # Beschäftigung und Erwerbstätigkeit - Struktur
    'Erwerbsquote',
    'Selbständigenquote',
    # Beschäftigung und Erwerbstätigkeit - Qualifikation
    'Beschäftigte ohne Berufsabschluss',
    'Beschäftigte mit Berufsabschluss',
    'Beschäftigte mit akademischem Berufsabschluss',
    'Beschäftigte mit Anforderungsniveau Experte',
    'Beschäftigte mit Anforderungsniveau Spezialist',
    'Beschäftigte mit Anforderungsniveau Fachkraft',
    'Beschäftigte mit Anforderungsniveau Helfer',
    # Beschäftigung und Erwerbstätigkeit - Wirtschafts- und Berufszweige
    # 'Beschäftigte Pimärer Sektor',
    # 'Beschäftigte Sekundärer Sektor',
    'Industriequote',
    # 'Beschäftigte Tertiärer Sektor',
    'Dienstleistungsquote',
    'Beschäftigte in unternehmensbezogenen Dienstleistungen',
    'Erwerbstätige Primärer Sektor',
    'Erwerbstätige Sekundärer Sektor',
    'Erwerbstätige Tertiärer Sektor',
    'Beschäftigte in Kreativbranchen',
    'Beschäftigte in wissensintensiven Industrien',
    'Anteil Erwerbstätige Verarbeitendes Gewerbe an Industrie',
    'Anteil Erwerbstätige Finanz- und Unternehmensdienstleistungen',
    # Bevölkerung - Bevölkerungsstruktur
    # 'Haushaltsgröße': n<100 -> take data from Zensus instead
    # 'Einpersonenhaushalte': n<100 -> take data from Zensus instead
    # Bildung – Ausbildungsangebot
    'Berufsschüler',
    # 'Berufsschülerinnen',
    # 'Ausländische Berufsschüler',
    'Ausbildungsplätze',
    'Auszubildende',
    'Weibliche Auszubildende',
    'Männliche Auszubildende',
    'Studierende',
    'Studierende an FH',
    # 'Weibliche Studierende',
    # 'Männliche Studierende',
    # 'Ausländische Studierende',
    # 'Studierende im 1. Semester',
    'Auszubildende je 100 Einwohner 15 bis 25 Jahre',
    'Studierende je 100 Einwohner 18 bis 25 Jahre',
    # Bildung - Schulische Bildung
    'Schüler',
    'Ausländische Schüler',
    'Ausländische Schüler je 100 Ausländer 6 bis 18 Jahre',
    'Schulabgänger mit Hauptschulabschluss',
    'Weibliche Schulabgänger mit Hauptschulabschluss',
    'Männliche Schulabgänger mit Hauptschulabschluss',
    'Schulabgänger mit mittlerem Abschluss',
    'Weibliche Schulabgänger mit mittlerem Abschluss',
    'Männliche Schulabgänger mit mittlerem Abschluss',
    'Schulabgänger mit allgemeiner Hochschulreife',
    # 'Weibliche Schulabgänger mit allgemeiner Hochschulreife',
    # 'Männliche Schulabgänger mit allgemeiner Hochschulreife',
    'Schulabgänger ohne Abschluss',
    'Weibliche Schulabgänger ohne Abschluss',
    'Männliche Schulabgänger ohne Abschluss',
    # Privateinkommen und private Schulden
    'Bruttoverdienst',
    'Bruttoverdienst im Produzierenden Gewerbe',
    'Medianeinkommen',
    'Medianeinkommen 25 bis unter 55-Jährige',
    'Medianeinkommen 55 bis unter 65-Jährige',
    'Haushaltseinkommen',
    'Verbraucherinsolvenzverfahren',
    'Schuldnerquote',
    'Arbeitsvolumen',
    'Medianeinkommen anerkannter Berufsabschluss',
    'Medianeinkommen akademischer Berufsabschluss',
    # Öffentliche Finanzen
    'Schlüsselzuweisungen',
    'Kommunale Schulden',
    'Kassenkredite',
    'Personal der Kommunen',
    'Ausgaben für Sachinvestitionen',
    'Zuweisungen für Investitionsfördermaßnahmen',
    # Raumwirksame Mittel
    'Städtebauförderung (langfristig)',
    'Städtebauförderung (kurzfristig)',
    # Siedlungsstruktur
    'Ländlichkeit',
    'Bevölkerung in Mittelzentren',
    'Bevölkerung in Oberzentren',
    # Sozialleistungen - Leistungsempfänger
    'SGB II - Quote',
    'Wohngeldhaushalte',
    'Wohngeldhaushalte (Mietzuschuss)',
    'Wohngeldhaushalte (Lastenzuschuss)',
    'Empfänger von Grundsicherung im Alter (Altersarmut)',
    'Empfänger von Mindestsicherungen',
    # Wirtschaft – Fremdenverkehr
    'Schlafgelegenheiten in Beherbergungsbetrieben',
    # Wirtschaft – Wirtschaftliche Leistung
    'Bruttoinlandsprodukt je Einwohner',
    'Bruttoinlandsprodukt je Erwerbstätigen',
    'Bruttowertschöpfung je Erwerbstätigen',
    'Bruttowertschöpfung je Erwerbstätigen Primärer Sektor',
    'Bruttowertschöpfung je Erwerbstätigen Sekundärer Sektor',
    'Bruttowertschöpfung je Erwerbstätigen Tertiärer Sektor',
    # 'Investitionen im Bergbau und Verarb. Gewerbe',
    # 'Auslandsumsatz im Bergbau u. Verarb. Gewerbe',
    'Umsatz im Bergbau u. Verarb. Gewerbe',
    'Umsatz Bauhauptgewerbe',
    'Anteil Bruttowertschöpfung Primärer Sektor',
    'Anteil Bruttowertschöpfung Sekundärer Sektor',
    'Anteil Bruttowertschöpfung Tertiärer Sektor'
]
# 2b. only available for other time periods
add_ind_c_dict = {
    # Beschäftigung und Erwerbstätigkeit – Wirtschafts- und Berufszweige
    'Beschäftigte im Handwerk': 2018,
    # Wirtschaft – Wirtschaftliche Leistung
    'Kleinstbetriebe': 2018,
    'Kleinbetriebe': 2018,
    'Mittlere Unternehmen': 2018,
    'Großunternehmen': 2018,
    'Umsatz im Handwerk': 2018
}

# variables needed to fix NaNs

ind_ma = ['Arbeitslose', 'Arbeitslose',
          'Neubauwohnungen je Einwohner', 'Neubauwohnungen je Einwohner',
          'Baugenehmigungen für Wohnungen', 'Baugenehmigungen für Wohnungen']
sub_ind_ma = ['Arbeitslose Frauen', 'Arbeitslose Männer',
              'Neue Ein- und Zweifamilienhäuser', 'Neubauwohnungen in Ein- und Zweifamilienhäusern',
              'Baugenehmigungen für Wohnungen in Ein- und Zweifamilienhäusern',
              'Baugenehmigungen für Wohnungen in Mehrfamilienhäusern']
dependencies_ind_ma = dict(zip(sub_ind_ma, ind_ma))

ind_c = ['Berufsschüler', 'Berufsschüler', 'Studierende', 'Studierende', 'Studierende', 'Studierende',
         'Umsatz im Bergbau u. Verarb. Gewerbe', 'Umsatz im Bergbau u. Verarb. Gewerbe',
         'Schulabgänger mit allgemeiner Hochschulreife', 'Schulabgänger mit allgemeiner Hochschulreife']
sub_ind_c = ['Berufsschülerinnen', 'Ausländische Berufsschüler', 'Weibliche Studierende', 'Männliche Studierende',
             'Ausländische Studierende', 'Studierende im 1. Semester', 'Auslandsumsatz im Bergbau u. Verarb. Gewerbe',
             'Investitionen im Bergbau und Verarb. Gewerbe', 'Weibliche Schulabgänger mit allgemeiner Hochschulreife',
             'Männliche Schulabgänger mit allgemeiner Hochschulreife']
dependencies_ind_c = dict(zip(sub_ind_c, ind_c))

# comments on indicators with NaNs 

comments_dict_missing_ma = {
    'sozialversicherungspflichtig Beschäftigte am Arbeitsort': 'small ma, prob. (close to) 0, no jobs',
    'Ein- und Zweiraumwohnungen': 'probably close to 0',
    'Einwohner-Arbeitsplatz-Dichte': 'prob. no jobs, = Einwohnerdichte?',
    'Einpendler': 'prob. no jobs',
    'Pendlersaldo': 'prob. no jobs',
    'Nahversorgung Haltestellen des ÖV Durchschnittsdistanz': 'prob. no bus stops',
    'Nahversorgung Haltestellen des ÖV Anteil der Bev. 1km Radius': 'prob. no bus stops'
}
comments_dict_missing_c = {'Industriequote': 'similar to other sector related ind.',
                           'Dienstleistungsquote': 'similar to other sector related ind.',
                           'Ausbildungsplätze': 'missing for Berlin',
                           'Ausgaben für Sachinvestitionen': 'same NaNs as Zuweisungen für Investitionsfördermaßnahmen',
                           'Zuweisungen für Investitionsfördermaßnahmen': 'same NaNs as Ausgaben für Sachinvestitionen',
                           'Bruttowertschöpfung je Erwerbstätigen Primärer Sektor': 'Bruttowertschöpfung im Prim. '
                                                                                    'Sektor approx. 0; no/ few workers '
                                                                                    'in this sector?'
                           }


########################### Regionalstatistik ##############################

# Columns of preprocessed election data from Regionalstatistik

col_id_regionalstatistik = 'code of spatial entity'
col_name_regionalstatistik = 'name of spatial entity'
col_eligible_voters = 'count eligible voters' #'Wahlberechtigte'
col_electoral_turnout = 'electoral turnout (percentage)' # 'Wahlbeteiligung'
col_vote_count_total = 'valid vote count' # Gültige Zweitstimmen'
col_vote_count_cdu = 'CDU/CSU'
col_vote_count_spd = 'SPD'
col_vote_count_green = 'The Greens' # 'Grüne'
col_vote_count_fdp = 'FDP'
col_vote_count_left = 'The Left' # 'Die Linke'
col_vote_count_afd = 'AfD'
col_vote_count_other = 'other parties' # 'Sonstige Parteien'

############################### Zensus ##########################################

col_name_zensus = 'name'

# name of feature owner occupation ratio
owner_occ_ratio = 'ownership occupation ratio'

# household size
col_count_households = 'count households'
col_count_one_p = 'share 1-person households'
col_count_two_p = 'share 2-person households'
col_count_three_p = 'share 3-person households'
col_count_four_p = 'share 4-person households'
col_count_five_p = 'share 5-person households'
col_count_six_more_p = 'share 6+-person households'

col_hh_sizes = [col_count_one_p,col_count_two_p,col_count_three_p,col_count_four_p,col_count_five_p,col_count_six_more_p]

########################### Radiation data ###################################
## Register of municipalities

col_longi = 'longitude'
col_lati = 'latitude'

# col_radiation = 'Global irradiation'
col_radiation = 'global radiation'

############################## PV data ##################################

list_relevant_features = ['EinheitMastrNummer','Gemeinde', 'Gemeindeschluessel', 'Registrierungsdatum',
                          'Inbetriebnahmedatum', 'EinheitSystemstatus', 'EinheitBetriebsstatus', 'Energietraeger',
                          'Nettonennleistung', 'AnzahlModule']
elem_name_xml_pv = 'EinheitSolar'
col_power_pv = 'peak power'
col_power_accum_pv = 'accumulated power'
# col_power_accum_pv = 'accumulated net nominal capacity'
col_com_date_pv = 'commissioning date' # Inbetriebnahmedatum
col_date_registration_pv = 'registration date'
col_installation_id_pv = 'id of pv system'
# col_installation_id = 'number of PV installations'
col_count_pv = 'number of PV installations'
col_count_pv_per_cap = col_count_pv + ' (per 1000 p)'
col_count_pv_accum = 'accumulated number of PV installations'
col_count_pv_accum_per_cap = col_count_pv_accum + ' (per 1000 p)'

# column in inkar data set: population size
var_pop_size = 'population'

# only consider registrations until 31/10/2022
t_reg_pv_latest_year = 2022
t_reg_pv_latest_month = 10

##################### Handling changes in AGS

col_destatis_type = 'type'
col_destatis_old_ags = 'old ags'
col_destatis_new_ags = 'new ags'
col_destatis_old_ars = 'old ars'
col_destatis_new_ars = 'new ars'
type_ma = 'Gemeindeverband'
type_m = 'Gemeinde'

## Recursive feature elimination and hyperparameter optimization
mean_r2_cv_train = 'mean r2 score cv (train)'
mean_r2_cv_test = 'mean r2 score cv (test)'
mean_mae_cv_train = 'mean mae cv (train)'
mean_mae_cv_test = 'mean mae cv (test)'
mean_mape_cv_train = 'mean mape cv (train)'
mean_mape_cv_test = 'mean mape cv (test)'
ranking_mean_r2_desc = 'rank_test_score_cv_descending'
col_id = 'simulation_id' # id of overall simulation
col_file_path = 'data_file'
col_idx_train = 'indices_training_set'
col_idx_val = 'indices_val_set'
col_idx_test = 'indices_test_set'
col_feature_count = 'count_features'
col_features = 'features'
col_target_feat = 'target_feature'
col_run_id = 'run'
col_mean_shap = 'mean shap'
col_std_shap = 'std shap'
col_occurences_feat = 'occurences feature'
col_predictions = 'predictions'

## parameters for lasso
col_alpha = 'alpha'
col_r2_train = 'r2 score training'
col_r2_test = 'r2 score test'
col_l2_train = 'l2 loss training'
col_l2_test = 'l2 loss test'
col_mse_train = 'mse training'
col_mse_test = 'mse test'
col_mae_train = 'mae training'
col_mae_test = 'mae test'
col_mape_train = 'mape training'
col_mape_test = 'mape test'
col_count_non_zero = 'number of features with non-zero coef.'
col_coef_one_norm = 'coefficients 1-norm'
col_mse_penalty_train = 'mse + coef. penalty training'
col_mse_penalty_test = 'mse + coef. penalty test'

col_abs_error = 'absolute error'
col_rel_error = 'relative error'
col_shap_other_parties = 'SHAP values other parties'

# only consider registrations until 31/10/2022
t_reg_pv_latest_year = 2023
t_reg_pv_latest_month = 9
min_power = [0]
max_power = [10]
max_nom_cap = 10
t_jan = 1
t_july = 7
t_sep = 9
t_dec = 12
t0_b = 1991
t0 = 2000
t1 = 2009
t2 = 2012
t3 = 2022
t4 = 2023

########################### Battery Electric Vehicles specific #################

col_bev_per_vehicle = 'priv. Elektro (BEV) per vehicle'

########################### Aggregation GemV ###################################
# Features that have to be summed when aggregatinging GemV
# Other are a population weighted average
name_of_aggregated_sum_features = ['unemployed',
                   'employees at place of work',
                   'employees at place of residence',
                   'population',
                   'male population',
                   'female population',
                   'population of working age',
                   'population (Census 2011)',
                   'floor space']

########################### Drop and normalize before run #######################

features_norm_drop_ls = ['male population', 'female population']

features_norm_to_population_ls = ['employees at place of work',
                         'employees at place of residence',
                         'unemployed', 'population of working age',]
                         
features_norm_to_population_bev_ls = ['chargingstations_before_2023'] 


#### Post to mattermost if run is finished

mattermost_url = None

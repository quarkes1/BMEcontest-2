# E01 Error Atlas

## Incumbent metrics

- TP / eligible / pred: 114 / 153 / 197
- FP / FN: 83 / 39

## FN stage decomposition

- FN-C0: 6
- FN-C1: 27
- FN-C2: 4
- FN-C3: 2

## Short FN breakdown

- long: 8
- medium: 12
- short: 19

## Short FN stage split

- FN-C0: 5
- FN-C1: 11
- FN-C2: 1
- FN-C3: 2

## FP taxonomy

- short_spike: 32
- long_continuous_activity: 28
- macro_high_micro_low: 4
- micro_high_macro_low: 18
- dense_cluster: 38
- isolated: 8
- same_truth_duplicate: 1

## Support and geometry

- macro_only: 4
- micro_only: 18
- macro_micro: 61

## FP Q3 distributions

- session_blend_percentile: {'count': 83, 'min': 0.5, 'p25': 1.0, 'median': 1.0, 'p75': 1.0, 'max': 1.0}
- candidate_score_percentile: {'count': 83, 'min': 0.5, 'p25': 1.0, 'median': 1.0, 'p75': 1.0, 'max': 1.0}
- duration_percentile: {'count': 83, 'min': 0.2, 'p25': 0.5, 'median': 0.75, 'p75': 1.0, 'max': 1.0}
- candidate_count_session: {'count': 83, 'min': 1.0, 'p25': 3.0, 'median': 4.0, 'p75': 5.0, 'max': 9.0}
- nearest_candidate_distance_ms: {'count': 82, 'min': 0.0, 'p25': 240000.0, 'median': 390000.0, 'p75': 525000.0, 'max': 48847500.0}

## FN cross statistics

- short: {'0': {'HNU21009': {'neither': 1}, 'HNU21010': {'macro-only': 1}, 'HNU21039': {'neither': 2, 'macro+micro': 1}}, '2': {'HNU21004': {'neither': 1, 'macro+micro': 1}, 'HNU21023': {'neither': 1}, 'HNU21014': {'micro-only': 1}, 'HNU212020': {'macro-only': 1}}, '3': {'HNU21025': {'macro-only': 3}, 'HNU21031': {'micro-only': 1, 'neither': 1}}, '4': {'HNU21008': {'macro-only': 1}, 'HNU21037': {'macro+micro': 1, 'micro-only': 2}}}
- medium: {'0': {'HNU21001': {'macro+micro': 1}, 'HNU21027': {'neither': 1}}, '2': {'HNU212020': {'macro-only': 1}, 'hnu21033': {'macro+micro': 2}}, '3': {'HNU21015': {'macro-only': 1}, 'HNU21025': {'micro-only': 1}}, '4': {'HNU21005': {'macro-only': 1, 'micro-only': 1}, 'HNU21002': {'macro+micro': 1}, 'HNU 21013': {'macro+micro': 1}, 'HNU21026': {'macro+micro': 1}}}
- long: {'1': {'HNU21032': {'macro+micro': 1}}, '2': {'HNU21012': {'micro-only': 2}, 'HNU21022': {'macro+micro': 1}, 'hnu21033': {'micro-only': 1}}, '4': {'HNU21002': {'macro+micro': 1}, 'HNU21005': {'micro-only': 1}, 'HNU21034': {'micro-only': 1}}}

## Subject concentration

- FP subject 6C7: 1
- FP subject HNU 21013: 1
- FP subject HNU21001: 2
- FP subject HNU21002: 1
- FP subject HNU21003: 2
- FP subject HNU21004: 1
- FP subject HNU21005: 3
- FP subject HNU21006: 3
- FP subject HNU21008: 2
- FP subject HNU21009: 2
- FP subject HNU21011: 1
- FP subject HNU21012: 4
- FP subject HNU21014: 3
- FP subject HNU21015: 2
- FP subject HNU21016: 4
- FP subject HNU21017: 1
- FP subject HNU21019: 3
- FP subject HNU21021: 1
- FP subject HNU21022: 4
- FP subject HNU21023: 2
- FP subject HNU21024: 3
- FP subject HNU21025: 1
- FP subject HNU21026: 1
- FP subject HNU21027: 2
- FP subject HNU21028: 2
- FP subject HNU21029: 1
- FP subject HNU21031: 2
- FP subject HNU21032: 3
- FP subject HNU21034: 3
- FP subject HNU21035: 5
- FP subject HNU21036: 2
- FP subject HNU21037: 1
- FP subject HNU21038: 3
- FP subject HNU21039: 5
- FP subject HNU21040: 2
- FP subject HNU21041: 3
- FP subject HNU212020: 1
- FN subject HNU 21013: 1
- FN subject HNU21001: 1
- FN subject HNU21002: 2
- FN subject HNU21004: 2
- FN subject HNU21005: 3
- FN subject HNU21008: 1
- FN subject HNU21009: 1
- FN subject HNU21010: 1
- FN subject HNU21012: 2
- FN subject HNU21014: 1
- FN subject HNU21015: 1
- FN subject HNU21022: 1
- FN subject HNU21023: 1
- FN subject HNU21025: 4
- FN subject HNU21026: 1
- FN subject HNU21027: 1
- FN subject HNU21031: 2
- FN subject HNU21032: 1
- FN subject HNU21034: 1
- FN subject HNU21037: 3
- FN subject HNU21039: 3
- FN subject HNU212020: 2
- FN subject hnu21033: 3

## Subject TP/FP/pred/eligible profile

- 6C7: TP=0, FP=1, pred=1, eligible=0
- HNU 21013: TP=2, FP=1, pred=3, eligible=3
- HNU21001: TP=4, FP=2, pred=6, eligible=5
- HNU21002: TP=4, FP=1, pred=5, eligible=6
- HNU21003: TP=2, FP=2, pred=4, eligible=2
- HNU21004: TP=3, FP=1, pred=4, eligible=5
- HNU21005: TP=3, FP=3, pred=6, eligible=6
- HNU21006: TP=5, FP=3, pred=8, eligible=5
- HNU21008: TP=4, FP=2, pred=6, eligible=5
- HNU21009: TP=4, FP=2, pred=6, eligible=5
- HNU21010: TP=1, FP=0, pred=1, eligible=2
- HNU21011: TP=5, FP=1, pred=6, eligible=5
- HNU21012: TP=0, FP=4, pred=4, eligible=2
- HNU21014: TP=3, FP=3, pred=6, eligible=4
- HNU21015: TP=4, FP=2, pred=6, eligible=5
- HNU21016: TP=1, FP=4, pred=5, eligible=1
- HNU21017: TP=5, FP=1, pred=6, eligible=5
- HNU21019: TP=3, FP=3, pred=6, eligible=3
- HNU21021: TP=4, FP=1, pred=5, eligible=4
- HNU21022: TP=0, FP=4, pred=4, eligible=1
- HNU21023: TP=4, FP=2, pred=6, eligible=5
- HNU21024: TP=3, FP=3, pred=6, eligible=3
- HNU21025: TP=4, FP=1, pred=5, eligible=8
- HNU21026: TP=5, FP=1, pred=6, eligible=6
- HNU21027: TP=0, FP=2, pred=2, eligible=1
- HNU21028: TP=2, FP=2, pred=4, eligible=2
- HNU21029: TP=4, FP=1, pred=5, eligible=4
- HNU21031: TP=4, FP=2, pred=6, eligible=6
- HNU21032: TP=4, FP=3, pred=7, eligible=5
- HNU21034: TP=3, FP=3, pred=6, eligible=4
- HNU21035: TP=2, FP=5, pred=7, eligible=2
- HNU21036: TP=6, FP=2, pred=8, eligible=6
- HNU21037: TP=2, FP=1, pred=3, eligible=5
- HNU21038: TP=3, FP=3, pred=6, eligible=3
- HNU21039: TP=3, FP=5, pred=8, eligible=6
- HNU21040: TP=3, FP=2, pred=5, eligible=3
- HNU21041: TP=1, FP=3, pred=4, eligible=1
- HNU212020: TP=2, FP=1, pred=3, eligible=4
- hnu21033: TP=2, FP=0, pred=2, eligible=5

## per-fold error profile

- fold 0: {'tp': 16, 'eligible': 23, 'pred': 35, 'fn': 7, 'fp': 19}
- fold 1: {'tp': 30, 'eligible': 31, 'pred': 46, 'fn': 1, 'fp': 16}
- fold 2: {'tp': 15, 'eligible': 27, 'pred': 33, 'fn': 12, 'fp': 18}
- fold 3: {'tp': 25, 'eligible': 32, 'pred': 42, 'fn': 7, 'fp': 17}
- fold 4: {'tp': 28, 'eligible': 40, 'pred': 41, 'fn': 12, 'fp': 13}

## Context-v1 residual separation

Descriptive FP-versus-TP means only; no classifier or learned transform was used.

## Q6 feature separation and sign consistency

- macro_pre_mean: global={'fp_mean': 0.1994939479931075, 'tp_mean': 0.17649738913944152, 'fp_minus_tp': 0.02299655885366597}; sign={'positive': 2, 'negative': 3, 'zero': 0, 'noncomparable': 0}
- macro_pre_max: global={'fp_mean': 0.399658647265263, 'tp_mean': 0.46538005471527943, 'fp_minus_tp': -0.06572140745001642}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- macro_pre_std: global={'fp_mean': 0.08000056432230283, 'tp_mean': 0.11079312525412964, 'fp_minus_tp': -0.030792560931826807}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- macro_pre_above_fraction: global={'fp_mean': 0.25544362132971005, 'tp_mean': 0.23785788486028334, 'fp_minus_tp': 0.01758573646942671}; sign={'positive': 3, 'negative': 2, 'zero': 0, 'noncomparable': 0}
- macro_candidate_mean: global={'fp_mean': 0.5165289140156063, 'tp_mean': 0.5861752673759556, 'fp_minus_tp': -0.0696463533603493}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- macro_candidate_max: global={'fp_mean': 0.7773269272672827, 'tp_mean': 0.8386433432706694, 'fp_minus_tp': -0.061316416003386665}; sign={'positive': 0, 'negative': 5, 'zero': 0, 'noncomparable': 0}
- macro_candidate_std: global={'fp_mean': 0.17799928238194038, 'tp_mean': 0.1878722938746984, 'fp_minus_tp': -0.009873011492758016}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- macro_candidate_above_fraction: global={'fp_mean': 0.772178974022686, 'tp_mean': 0.8450954922283668, 'fp_minus_tp': -0.07291651820568079}; sign={'positive': 2, 'negative': 3, 'zero': 0, 'noncomparable': 0}
- macro_post_mean: global={'fp_mean': 0.22078418225709748, 'tp_mean': 0.2327601736396244, 'fp_minus_tp': -0.011975991382526918}; sign={'positive': 2, 'negative': 3, 'zero': 0, 'noncomparable': 0}
- macro_post_max: global={'fp_mean': 0.5211516600747708, 'tp_mean': 0.6061877847285639, 'fp_minus_tp': -0.08503612465379318}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- macro_post_std: global={'fp_mean': 0.1336616222970064, 'tp_mean': 0.1520769688552797, 'fp_minus_tp': -0.018415346558273288}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- macro_post_above_fraction: global={'fp_mean': 0.2875441781530482, 'tp_mean': 0.3327683877996954, 'fp_minus_tp': -0.045224209646647195}; sign={'positive': 2, 'negative': 3, 'zero': 0, 'noncomparable': 0}
- macro_candidate_minus_pre_mean: global={'fp_mean': 0.30391519826722246, 'tp_mean': 0.40964594214964867, 'fp_minus_tp': -0.1057307438824262}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- macro_candidate_minus_pre_max: global={'fp_mean': 0.3842835476899852, 'tp_mean': 0.3744276961954128, 'fp_minus_tp': 0.00985585149457241}; sign={'positive': 3, 'negative': 2, 'zero': 0, 'noncomparable': 0}
- macro_candidate_minus_pre_std: global={'fp_mean': 0.11246663539190316, 'tp_mean': 0.07831515435335804, 'fp_minus_tp': 0.03415148103854512}; sign={'positive': 4, 'negative': 1, 'zero': 0, 'noncomparable': 0}
- macro_candidate_minus_pre_above_fraction: global={'fp_mean': 0.5075639197121646, 'tp_mean': 0.6058667710161221, 'fp_minus_tp': -0.09830285130395744}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- macro_candidate_minus_post_mean: global={'fp_mean': 0.28987114697895244, 'tp_mean': 0.3534150937363312, 'fp_minus_tp': -0.06354394675737873}; sign={'positive': 0, 'negative': 5, 'zero': 0, 'noncomparable': 0}
- macro_candidate_minus_post_max: global={'fp_mean': 0.2546060623548755, 'tp_mean': 0.23245555854210545, 'fp_minus_tp': 0.022150503812770034}; sign={'positive': 4, 'negative': 1, 'zero': 0, 'noncomparable': 0}
- macro_candidate_minus_post_std: global={'fp_mean': 0.04679507995497536, 'tp_mean': 0.035795325019418685, 'fp_minus_tp': 0.010999754935556674}; sign={'positive': 3, 'negative': 2, 'zero': 0, 'noncomparable': 0}
- macro_candidate_minus_post_above_fraction: global={'fp_mean': 0.4761358900530212, 'tp_mean': 0.5123271044286714, 'fp_minus_tp': -0.03619121437565015}; sign={'positive': 2, 'negative': 3, 'zero': 0, 'noncomparable': 0}
- macro_pre_coverage: global={'fp_mean': 0.632658573452652, 'tp_mean': 0.7113519754678124, 'fp_minus_tp': -0.0786934020151604}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- macro_candidate_coverage: global={'fp_mean': 0.9135497071241349, 'tp_mean': 0.9553724217904908, 'fp_minus_tp': -0.0418227146663559}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- macro_post_coverage: global={'fp_mean': 0.5595936424658159, 'tp_mean': 0.5677948345989576, 'fp_minus_tp': -0.008201192133141744}; sign={'positive': 3, 'negative': 2, 'zero': 0, 'noncomparable': 0}
- macro_neighbor_run_count: global={'fp_mean': 1.1204819277108433, 'tp_mean': 1.3157894736842106, 'fp_minus_tp': -0.1953075459733673}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- macro_neighbor_run_total_duration_s: global={'fp_mean': 474.2168674698795, 'tp_mean': 539.078947368421, 'fp_minus_tp': -64.86207989854154}; sign={'positive': 2, 'negative': 3, 'zero': 0, 'noncomparable': 0}
- macro_neighbor_run_max_duration_s: global={'fp_mean': 484.1666666666667, 'tp_mean': 482.5301204819277, 'fp_minus_tp': 1.636546184738961}; sign={'positive': 3, 'negative': 2, 'zero': 0, 'noncomparable': 0}
- macro_preceding_run_distance_s: global={'fp_mean': 422.77777777777777, 'tp_mean': 407.3529411764706, 'fp_minus_tp': 15.424836601307163}; sign={'positive': 4, 'negative': 1, 'zero': 0, 'noncomparable': 0}
- macro_following_run_distance_s: global={'fp_mean': 473.46153846153845, 'tp_mean': 524.2622950819672, 'fp_minus_tp': -50.80075662042873}; sign={'positive': 2, 'negative': 3, 'zero': 0, 'noncomparable': 0}
- macro_candidate_center_half_mass_fraction: global={'fp_mean': 0.6318599098990751, 'tp_mean': 0.5807056818420672, 'fp_minus_tp': 0.05115422805700798}; sign={'positive': 4, 'negative': 1, 'zero': 0, 'noncomparable': 0}
- macro_candidate_first_minus_second_mean: global={'fp_mean': -0.02790194580549053, 'tp_mean': -0.053332639345521106, 'fp_minus_tp': 0.025430693540030577}; sign={'positive': 4, 'negative': 1, 'zero': 0, 'noncomparable': 0}
- micro_pre_mean: global={'fp_mean': 0.3313434296517348, 'tp_mean': 0.30992867288150117, 'fp_minus_tp': 0.021414756770233634}; sign={'positive': 2, 'negative': 3, 'zero': 0, 'noncomparable': 0}
- micro_pre_max: global={'fp_mean': 0.8131915215572043, 'tp_mean': 0.8318436132108695, 'fp_minus_tp': -0.018652091653665215}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- micro_pre_std: global={'fp_mean': 0.19772829814006157, 'tp_mean': 0.21960802529751677, 'fp_minus_tp': -0.021879727157455203}; sign={'positive': 0, 'negative': 5, 'zero': 0, 'noncomparable': 0}
- micro_pre_above_fraction: global={'fp_mean': 0.5735287889528868, 'tp_mean': 0.5508949258598278, 'fp_minus_tp': 0.022633863093059037}; sign={'positive': 2, 'negative': 3, 'zero': 0, 'noncomparable': 0}
- micro_candidate_mean: global={'fp_mean': 0.5849146508262009, 'tp_mean': 0.6157689263700402, 'fp_minus_tp': -0.03085427554383935}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- micro_candidate_max: global={'fp_mean': 0.9087329602189734, 'tp_mean': 0.9215301667112621, 'fp_minus_tp': -0.01279720649228866}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- micro_candidate_std: global={'fp_mean': 0.20931613446731978, 'tp_mean': 0.20730424628588665, 'fp_minus_tp': 0.0020118881814331357}; sign={'positive': 2, 'negative': 3, 'zero': 0, 'noncomparable': 0}
- micro_candidate_above_fraction: global={'fp_mean': 0.9089232589211764, 'tp_mean': 0.9292879428814693, 'fp_minus_tp': -0.020364683960292895}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- micro_post_mean: global={'fp_mean': 0.3478709052516142, 'tp_mean': 0.34104176160075783, 'fp_minus_tp': 0.006829143650856351}; sign={'positive': 3, 'negative': 2, 'zero': 0, 'noncomparable': 0}
- micro_post_max: global={'fp_mean': 0.8334935322383339, 'tp_mean': 0.8693409468567314, 'fp_minus_tp': -0.035847414618397466}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- micro_post_std: global={'fp_mean': 0.22002533436607308, 'tp_mean': 0.23299952083065273, 'fp_minus_tp': -0.012974186464579651}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- micro_post_above_fraction: global={'fp_mean': 0.6016612179967449, 'tp_mean': 0.5973839699989879, 'fp_minus_tp': 0.004277247997756972}; sign={'positive': 2, 'negative': 3, 'zero': 0, 'noncomparable': 0}
- micro_candidate_minus_pre_mean: global={'fp_mean': 0.2442997502242611, 'tp_mean': 0.30647941684885754, 'fp_minus_tp': -0.06217966662459645}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- micro_candidate_minus_pre_max: global={'fp_mean': 0.09313150835127539, 'tp_mean': 0.08977927849587822, 'fp_minus_tp': 0.003352229855397168}; sign={'positive': 4, 'negative': 1, 'zero': 0, 'noncomparable': 0}
- micro_candidate_minus_pre_std: global={'fp_mean': 0.015628049645035984, 'tp_mean': -0.012369025658913999, 'fp_minus_tp': 0.027997075303949983}; sign={'positive': 5, 'negative': 0, 'zero': 0, 'noncomparable': 0}
- micro_candidate_minus_pre_above_fraction: global={'fp_mean': 0.32767886736550117, 'tp_mean': 0.37823301279003835, 'fp_minus_tp': -0.05055414542453718}; sign={'positive': 2, 'negative': 3, 'zero': 0, 'noncomparable': 0}
- micro_candidate_minus_post_mean: global={'fp_mean': 0.23649008271966174, 'tp_mean': 0.2747271647692824, 'fp_minus_tp': -0.038237082049620646}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- micro_candidate_minus_post_max: global={'fp_mean': 0.0754551066662358, 'tp_mean': 0.05218921985453078, 'fp_minus_tp': 0.023265886811705015}; sign={'positive': 4, 'negative': 1, 'zero': 0, 'noncomparable': 0}
- micro_candidate_minus_post_std: global={'fp_mean': -0.00971385499200041, 'tp_mean': -0.025695274544766087, 'fp_minus_tp': 0.015981419552765677}; sign={'positive': 4, 'negative': 1, 'zero': 0, 'noncomparable': 0}
- micro_candidate_minus_post_above_fraction: global={'fp_mean': 0.3058531999786499, 'tp_mean': 0.33190397288248136, 'fp_minus_tp': -0.026050772903831454}; sign={'positive': 2, 'negative': 3, 'zero': 0, 'noncomparable': 0}
- micro_pre_coverage: global={'fp_mean': 0.806328456155531, 'tp_mean': 0.8986755602325843, 'fp_minus_tp': -0.09234710407705338}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- micro_candidate_coverage: global={'fp_mean': 0.9962097618009804, 'tp_mean': 0.9977986445906362, 'fp_minus_tp': -0.0015888827896557434}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- micro_post_coverage: global={'fp_mean': 0.7256147911881915, 'tp_mean': 0.7773664093381981, 'fp_minus_tp': -0.05175161815000662}; sign={'positive': 0, 'negative': 5, 'zero': 0, 'noncomparable': 0}
- micro_neighbor_run_count: global={'fp_mean': 3.6144578313253013, 'tp_mean': 3.956140350877193, 'fp_minus_tp': -0.3416825195518918}; sign={'positive': 0, 'negative': 5, 'zero': 0, 'noncomparable': 0}
- micro_neighbor_run_total_duration_s: global={'fp_mean': 787.7710843373494, 'tp_mean': 973.5526315789474, 'fp_minus_tp': -185.781547241598}; sign={'positive': 1, 'negative': 4, 'zero': 0, 'noncomparable': 0}
- micro_neighbor_run_max_duration_s: global={'fp_mean': 478.21875, 'tp_mean': 561.1184210526316, 'fp_minus_tp': -82.89967105263156}; sign={'positive': 2, 'negative': 3, 'zero': 0, 'noncomparable': 0}
- micro_preceding_run_distance_s: global={'fp_mean': 238.72641509433961, 'tp_mean': 249.08333333333334, 'fp_minus_tp': -10.356918238993728}; sign={'positive': 2, 'negative': 3, 'zero': 0, 'noncomparable': 0}
- micro_following_run_distance_s: global={'fp_mean': 289.36567164179104, 'tp_mean': 300.66964285714283, 'fp_minus_tp': -11.303971215351794}; sign={'positive': 3, 'negative': 2, 'zero': 0, 'noncomparable': 0}
- micro_candidate_center_half_mass_fraction: global={'fp_mean': 0.529096438417179, 'tp_mean': 0.5291358549241184, 'fp_minus_tp': -3.941650693939369e-05}; sign={'positive': 2, 'negative': 3, 'zero': 0, 'noncomparable': 0}
- micro_candidate_first_minus_second_mean: global={'fp_mean': -0.02593746075661174, 'tp_mean': -0.0488638395886251, 'fp_minus_tp': 0.022926378832013358}; sign={'positive': 3, 'negative': 2, 'zero': 0, 'noncomparable': 0}

## Recommended next experiments

Evidence-based mechanisms only; no outer-derived feature selection was performed.
- candidate_recall: observed=27 FN-C1 of 39 FN; short-stage split={'FN-C0': 5, 'FN-C1': 11, 'FN-C2': 1, 'FN-C3': 2}; recommendation=Prioritize candidate-control reachability diagnostics before changing verifier complexity.
- scale_support: observed={'macro+micro': 12, 'macro-only': 9, 'micro-only': 11, 'neither': 7}; recommendation=Compare macro-only, micro-only and joint support in a preregistered candidate experiment; this atlas does not select features.
- false_positive_concentration: observed=83 unmatched final predictions across 39 subjects; recommendation=Use subject/session-local diagnostics to formulate, but not fit, a later hard-negative hypothesis.

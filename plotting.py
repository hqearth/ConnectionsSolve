import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

losses = [0.0646063, 0.062187657, 0.065627515, 0.063344896, 0.061465748, 0.059861332, 0.062146552, 0.060722332, 0.05986827, 0.05729709, 0.056192435, 0.056267414, 0.05364085, 0.054078627, 0.053613596, 0.0541981, 0.05471701, 0.054173477, 0.06364804, 0.06489934, 0.06401647, 0.06858861, 0.06829514, 0.0683727, 0.067182265, 0.06901794, 0.06759005, 0.06665851, 0.06555318, 0.06434218, 0.06515939, 0.064168006, 0.06398244, 0.06328135, 0.06359166, 0.063017964, 0.062482752, 0.061603446, 0.06208974, 0.06257351, 0.06220232, 0.06145051, 0.06134359, 0.060939755, 0.06021723, 0.059969414, 0.059222855, 0.058829445, 0.058298998, 0.060356416, 0.06030393, 0.059698198, 0.059585333, 0.058980823, 0.058927856, 0.05844822, 0.057783805, 0.057542212, 0.057296883, 0.05699784, 0.056269024, 0.055633545, 0.055458326, 0.0549671, 0.05502462, 0.05470557, 0.05406701, 0.05358532, 0.053231396, 0.053364273, 0.05320721, 0.05316786, 0.052804668, 0.052625164, 0.05279224, 0.05306089, 0.052909046, 0.052768953, 0.052585755, 0.052465033, 0.052442797, 0.05254647, 0.05238271, 0.05211799, 0.052004285, 0.05218313, 0.052189723, 0.052346025, 0.052148696, 0.05200346, 0.05193362, 0.051844344, 0.051488485, 0.05141532, 0.051282417, 0.05150947, 0.0513054, 0.051240828, 0.05111282, 0.050897054, 0.050702084, 0.050948255, 0.051330913, 0.051048916, 0.050888076, 0.05067423, 0.050463486, 0.050405312, 0.050661, 0.05037029, 0.050377626, 0.05010124, 0.049853202, 0.049785405, 0.049696203, 0.04959657, 0.049526606, 0.049399834, 0.049219664, 0.048978884, 0.04888397, 0.048800774, 0.048756905, 0.04862687, 0.04855617, 0.048575986, 0.048524354, 0.04838445, 0.04834587, 0.048370562, 0.048240874, 0.048123676, 0.048063666, 0.048033908, 0.04789348, 0.0477943, 0.047755297, 0.04775984, 0.04772178, 0.047830846, 0.04778469, 0.04764729, 0.04741753, 0.047265258, 0.047249276, 0.047569577, 0.04755056, 0.047516476, 0.04763891, 0.04753221, 0.047533255, 0.047513925, 0.047431275, 0.04737485, 0.04747374, 0.047592975, 0.047749095, 0.047794763, 0.047730517, 0.047613643, 0.047764417, 0.047735427, 0.047954403, 0.047822602, 0.047746137, 0.04856047, 0.048645414, 0.04856608, 0.048452467, 0.048404884, 0.04835606, 0.048312325, 0.048491675, 0.048421595, 0.048390634, 0.048371065, 0.048367098, 0.04845145, 0.048647296, 0.048547767, 0.048632815, 0.04861551, 0.04903263, 0.04911082, 0.049094513, 0.049014665, 0.048984524, 0.048870128, 0.04893079, 0.048729926, 0.04872164, 0.048803687, 0.04875337, 0.048833206, 0.04873603, 0.048676267, 0.048559368, 0.04858747, 0.048474163, 0.048610963, 0.04855746, 0.048629705, 0.04847447, 0.04840003, 0.048407968, 0.048450857, 0.04875193, 0.048704837, 0.048842944, 0.048745647, 0.048651837, 0.048811983, 0.048837025, 0.04885177, 0.04892931, 0.048891433, 0.04882869, 0.048774067, 0.048697315, 0.048685182, 0.04881365, 0.048641574, 0.048611723, 0.048609514, 0.048854437, 0.048816454, 0.04893208, 0.048988454, 0.048955586, 0.04889813, 0.04874761, 0.048796006, 0.048992783, 0.049115457, 0.049110044, 0.049195644, 0.049166337, 0.049033068, 0.049020782, 0.04891606, 0.048857454, 0.04874942, 0.048775394, 0.04908674, 0.049118407, 0.0492876, 0.050127964, 0.050059803, 0.05017884, 0.05041931, 0.05045242, 0.05098433, 0.051317237, 0.051363517, 0.05159408, 0.051574446, 0.051995806, 0.052156158, 0.052148018, 0.052184127, 0.052145682, 0.052933164, 0.052954387, 0.053099748, 0.053175237, 0.053121917, 0.053266294, 0.053339187, 0.05334471, 0.053400483, 0.05364757, 0.053734504, 0.053776912, 0.053864032, 0.053829152, 0.053834688, 0.053702086, 0.053743828, 0.05393725, 0.05390863, 0.05387328, 0.05404886, 0.05446568, 0.05487675, 0.055166125, 0.055157978, 0.05520489, 0.05512555, 0.05511635, 0.055154115, 0.055164207, 0.05515429, 0.05519759, 0.055174574, 0.055166498, 0.055366002, 0.05545036, 0.05544053, 0.055618923, 0.055691645, 0.05573829, 0.055753883, 0.05570188, 0.055844318, 0.056110334, 0.056064263, 0.056251675, 0.056213405, 0.056169473, 0.056961935, 0.05741847, 0.057392757, 0.05770261, 0.057806574, 0.057712283, 0.057713848, 0.057809066, 0.05770819, 0.057761226, 0.057721175, 0.057684314, 0.05786333, 0.0579111, 0.057994854, 0.05818172, 0.058381006, 0.058406245, 0.05827793, 0.058400042, 0.058342714, 0.058209267, 0.058310807, 0.058419786, 0.058630742, 0.058516636, 0.05860036, 0.058561914, 0.05853969, 0.058556173, 0.05854965, 0.05860083, 0.058605764, 0.05854232, 0.05864687, 0.05886241, 0.0587461, 0.05876667, 0.058767613, 0.058837894, 0.05884452, 0.05887545, 0.058895353, 0.058876697, 0.05897815, 0.059038002, 0.05918064, 0.05910968, 0.059165325, 0.059184875, 0.0590942, 0.059006255, 0.058901444, 0.058816653, 0.05880268, 0.05879663, 0.058684353, 0.058693934, 0.059028983, 0.0589289, 0.059119776, 0.05910022, 0.05918319, 0.059249192, 0.05921685, 0.05932003, 0.05942967, 0.059333593, 0.059250765, 0.05925583, 0.059425715, 0.059355523, 0.05927321, 0.059307937, 0.05942453, 0.05964682, 0.05955144, 0.059657805, 0.059720974, 0.059952997, 0.060006812, 0.060121015, 0.060397357, 0.060432497, 0.060458653, 0.060586162, 0.060538013, 0.06052717, 0.06058044, 0.06050911, 0.06065749, 0.06068184, 0.060724966, 0.060840953, 0.060715318, 0.060740568, 0.06086789, 0.060849138, 0.060844533, 0.060721282, 0.060999893, 0.06101314, 0.06112574, 0.061351683, 0.061361626, 0.061549783, 0.061672628, 0.06168083, 0.061738968, 0.061737232, 0.06174347, 0.061807774, 0.061836112, 0.06194996, 0.061892893, 0.062088754, 0.061991833, 0.062347967, 0.06230713, 0.062334895, 0.062463395, 0.0624777, 0.062445905, 0.06235128, 0.06236648, 0.06251899, 0.06244375, 0.06251323, 0.0625587, 0.06259797, 0.06266292, 0.06261606, 0.06263465, 0.06284522, 0.06288648, 0.062991686, 0.0631407, 0.063226305, 0.0632621, 0.06335523, 0.06341287, 0.063665055, 0.06366106, 0.063687466, 0.06377964, 0.06376364, 0.063837826, 0.063982695, 0.06430814, 0.06429497, 0.06446006, 0.06439072, 0.064383455, 0.06445086, 0.06448836, 0.06458828, 0.064696565, 0.06472167, 0.06477804, 0.06481898, 0.06492264, 0.064990096, 0.06505765, 0.06516254, 0.06525797, 0.065382205, 0.065428115, 0.06560452, 0.06570436, 0.065792695, 0.065885015, 0.06604384, 0.066132635, 0.06615491, 0.06622984, 0.066324815, 0.066473454, 0.066607766, 0.06667873, 0.066773385, 0.066990435, 0.067129016, 0.06733437, 0.06746379, 0.067596875, 0.0677667, 0.06790864, 0.0680424, 0.06834217, 0.06847339, 0.06870076, 0.068881564, 0.06899527, 0.06916081, 0.06935064, 0.06954205, 0.069679745, 0.06979283, 0.06998929, 0.07050053, 0.070662856, 0.07091253, 0.071062, 0.071273, 0.071435384, 0.07172106, 0.071994, 0.07221306, 0.07242456, 0.072564036, 0.07285284, 0.07317014, 0.073361814, 0.07361239, 0.073834546, 0.07400407, 0.07422241, 0.0743883, 0.074622504, 0.07476908, 0.07500617, 0.07524666, 0.07548489, 0.07583286, 0.07605119, 0.07640278, 0.07656958, 0.0768699, 0.07726618, 0.07754248, 0.077772126, 0.078026496, 0.07829753, 0.078516275, 0.07875257, 0.078971915, 0.07915256, 0.07947235, 0.079720765, 0.07985874, 0.080145225, 0.08032642, 0.080621906, 0.08087938, 0.08105332, 0.0813258, 0.08162911, 0.08186774, 0.082177855, 0.08246108, 0.08287402, 0.08323996, 0.08352907, 0.08380675, 0.084079646, 0.08467058, 0.0849689, 0.08517547, 0.08547872, 0.085774735, 0.08607842, 0.086353526, 0.08665945, 0.08713029, 0.08735682, 0.08759061, 0.08783303, 0.08820974, 0.08858523, 0.08901763, 0.089317165, 0.08960193, 0.08998365, 0.0902187, 0.09054108, 0.090896316, 0.091093004, 0.09130924, 0.09156442, 0.09194113, 0.09226314, 0.09243741, 0.09275005, 0.09306203, 0.093304396, 0.09355956, 0.09388859, 0.094114214, 0.09432459, 0.09451325, 0.09464509, 0.094949394, 0.09514459, 0.09554251, 0.09576987, 0.09586985, 0.09609614, 0.09625194, 0.09660263, 0.09681489, 0.09694068, 0.09724492, 0.09750635, 0.09772671, 0.09799031, 0.09808741, 0.0983661, 0.09861488, 0.09865225, 0.09896482, 0.09979464, 0.10000216, 0.100228444, 0.100322686, 0.100551195, 0.10065378, 0.100927725, 0.10110821, 0.10102516, 0.101169944, 0.101536386, 0.10146877, 0.101864465, 0.102063954, 0.10238527, 0.10274963, 0.102826916, 0.10328526, 0.10344957, 0.10350875, 0.10346758, 0.10346294, 0.103378415, 0.103804216, 0.10399242, 0.10420934, 0.10451198, 0.10466661, 0.10486838, 0.10496173, 0.10510935, 0.10536773, 0.10538522, 0.10529775, 0.10532804, 0.105396815, 0.10537794, 0.105476834, 0.10558864, 0.1058351, 0.10591128, 0.10593595, 0.10606325, 0.10617908, 0.106414475, 0.106433555, 0.10684464, 0.10700681, 0.10694281, 0.106962346, 0.1069412, 0.10701082, 0.1071343, 0.1072734, 0.107291855, 0.1073176, 0.107357375, 0.10746525, 0.10750558, 0.10757236, 0.10760231, 0.10761023, 0.107543126, 0.10768476, 0.10787655, 0.10791873, 0.107981354, 0.10793011, 0.10797171, 0.108019374, 0.10802935, 0.10809877, 0.108204745, 0.10824253, 0.1082311, 0.108224995, 0.108297005, 0.10835296, 0.10841719, 0.10844487, 0.108547494, 0.10863707, 0.1087137, 0.10875625, 0.108766295, 0.10884053, 0.10893619, 0.108949326, 0.10894992, 0.108970225, 0.10913864, 0.109245054, 0.10929679, 0.10927676, 0.10934212, 0.109487094, 0.10952865, 0.10963196, 0.10969648, 0.10971759, 0.10970425, 0.10983183, 0.1099175, 0.11001683, 0.11007972, 0.11006877, 0.11018733, 0.11026179, 0.11025128, 0.11016427, 0.11011944, 0.110211134, 0.11021996, 0.11022536, 0.11029559, 0.11024976, 0.110233314, 0.11038867, 0.11050197, 0.11060119, 0.11059927, 0.11056454, 0.110544704, 0.1106388, 0.11075263, 0.110667236, 0.11084114, 0.11077023, 0.11075413, 0.111042485, 0.11106892, 0.110955186, 0.11099405, 0.111019604, 0.11105889, 0.11134521, 0.11144642, 0.11156581, 0.111607835, 0.11167252, 0.11187037, 0.11197323, 0.112211294, 0.11226944, 0.11239918, 0.112544656, 0.11259294, 0.11275082, 0.11281973, 0.11294718, 0.113096975, 0.11312025, 0.11317663, 0.11310012, 0.113197386, 0.113148846, 0.113276154, 0.11338722, 0.113405146, 0.11357173, 0.11368664, 0.113646075]
avg_r = [-8.051724172924995, -7.514223481764111, -7.690143851364737, -7.446871976950617, -7.346910639703647, -7.004224344352821, -7.3670683580692335, -7.362007816413357, -7.541397286654068, -7.435222729797649, -7.047592577079768, -7.004686955174673, -7.150425287156343, -6.847940246941148, -6.738351527955293, -6.796790506801809, -6.942463628849642, -6.938517182723851, -6.933226425990368, -6.802477420583903, -6.8785067327262235, -6.9110192160993345, -6.877743681840665, -6.868690787653288, -6.8709937809109976, -6.892582322997402, -6.944529604885285, -7.018901688800956, -7.078523140796675, -7.088129119541362, -7.07811335311386, -7.147272139424664, -7.223336870757986, -7.330010741736796, -7.369753150480717, -7.389529935315808, -7.442856120783537, -7.491106041737612, -7.478557982191084, -7.4626009067960215, -7.458501651434675, -7.414292803669694, -7.421176498346802, -7.473311973237771, -7.45008596308941, -7.437176743388776, -7.4605720853525055, -7.495260263509842, -7.50678693427964, -7.482954361669432, -7.474991445061679, -7.450537844497985, -7.4820760724621325, -7.5486878559499315, -7.567473420650786, -7.562093731267694, -7.584808197386249, -7.581121985814646, -7.578388991154122, -7.592884446473386, -7.546411975406702, -7.609144585147953, -7.638407040873336, -7.645270062137543, -7.652307080911592, -7.654300528909894, -7.661879266639235, -7.666831806499994, -7.63442814409835, -7.671541640604593, -7.666106056706248, -7.611108276627219, -7.608531693895527, -7.578974561815462, -7.585467075367038, -7.563890300100186, -7.591520084088842, -7.567477036304039, -7.571969354550502, -7.57399753708979, -7.558044519148644, -7.5348864471087955, -7.523069183659131, -7.530650920024111, -7.565220383036165, -7.511185973746219, -7.521526168359448, -7.519022367896445, -7.525774014767124, -7.482941088838968, -7.494251383061976, -7.493575099455483, -7.472722248351391, -7.47291623427456, -7.43890401608788, -7.453965183228746, -7.437973814454425, -7.4462909283799, -7.431170970802003, -7.44521612524446, -7.446833611006688, -7.416156549468808, -7.373497353975494, -7.388765273595719, -7.374135109918805, -7.382548499765305, -7.349223508429324, -7.3677511487804805, -7.34146367303797, -7.319982624173541, -7.3125186709262495, -7.289719595056117, -7.31730091496995, -7.303644376264992, -7.299601780280351, -7.289471929902885, -7.271258944416358, -7.283790342003397, -7.2705097372583305, -7.25029650214125, -7.210421465559258, -7.168487672107683, -7.152704594048514, -7.162335237258473, -7.175705035523127, -7.177856479545389, -7.1920172161244365, -7.182469345485753, -7.219221659744753, -7.227420732333601, -7.2042843123755596, -7.183132963964781, -7.1803536215780195, -7.200695109771396, -7.183482048005548, -7.180711603617467, -7.19346771032135, -7.205044557734202, -7.174416279312545, -7.141861814046483, -7.147634526326145, -7.153439838018095, -7.100044414250276, -7.089106305926788, -7.07489131218541, -7.058642602897342, -7.063713498354412, -7.073026548032212, -7.070171089161609, -7.063571936750451, -7.0797768663882525, -7.084742250148727, -7.075613796282329, -7.071028100876382, -7.059298719506519, -7.057495716033607, -7.030443765267622, -7.0284461919346395, -7.044230575892187, -7.050527659345962, -7.034884123911821, -7.038704246334481, -7.0134433310483635, -7.065451658588445, -7.050187394189452, -7.0238935982019965, -7.039509222790457, -7.040980395113889, -7.021568414552445, -7.022035807439481, -7.0270217976883815, -7.021405498233197, -7.028971947985834, -7.014433058561045, -7.017814072420281, -7.211495956117473, -7.192440826782398, -7.186818809875077, -7.151423843585439, -7.149697828292561, -7.097718029652405, -7.114395920807353, -7.115471419818737, -7.130900621618926, -7.136330987612719, -7.128527447002004, -7.131172267998655, -7.125874237400462, -7.108910667735693, -7.069988072419268, -7.095431751060623, -7.0983466534846995, -7.072586143024281, -7.05108657658353, -7.052020096055315, -7.05976555554616, -7.047722401297067, -7.052713598096298, -7.090117711006833, -7.0732282153647725, -7.06334958107953, -7.066394868383351, -7.052689800576883, -7.036225888019876, -7.026072324439932, -7.032478151329016, -7.0015940846339735, -6.985617930147923, -6.935531831751017, -6.911261787053647, -6.918914798472861, -6.8670793059044515, -6.887454701105421, -6.898128902861371, -6.909299626501977, -6.9013008684210115, -6.899743177510521, -6.904215686880412, -6.896274393749809, -6.904218134507903, -6.9353999749384245, -6.890312609913991, -6.878949761359466, -6.823856693525459, -6.797870814403003, -6.793783401070504, -6.805116302868863, -6.81161906554402, -6.843776462912617, -6.825615880561885, -6.796755609240238, -6.785106609247852, -6.796241279537561, -6.782802223859259, -6.785863146721337, -6.794068348771017, -6.806865394583626, -6.8184105864289535, -6.791232799738487, -6.805361447117035, -6.759342786401391, -6.750743111079373, -6.789690154587492, -6.793344771773485, -6.854718155156806, -6.879933631519333, -6.85949510247996, -6.8478124298842, -6.938242923458886, -6.999775612343086, -6.998935810437965, -6.98308913709854, -7.00782831840485, -7.055998773315127, -7.042647809644648, -7.029646285663682, -7.023868650345173, -7.002510681398305, -6.981298413431866, -6.9776794619551366, -6.960635953492974, -6.949812330074686, -6.940417593270141, -6.941596320958078, -6.9366737093477076, -6.936886444519625, -6.929839679624313, -6.934562734661056, -6.926014670193019, -6.914352068251449, -6.90360130248076, -6.891088528532534, -6.892706556842579, -6.874479049130949, -6.862017952523096, -6.856179478108167, -6.872295892894902, -6.869949753771957, -6.809689340180329, -6.78275838981094, -6.7929984459992925, -6.772120509420972, -6.711436042439063, -6.6842350800257435, -6.654929592490719, -6.6560226818783725, -6.637577238952269, -6.630309720169121, -6.624664354017389, -6.623732443197533, -6.623204632253614, -6.616532524962732, -6.616058636020615, -6.600803969055519, -6.598132049496115, -6.58640795379662, -6.575575864845165, -6.562533484995224, -6.50323564266769, -6.4835090341599475, -6.480886531882164, -6.4237644623642085, -6.407060145567498, -6.4090943054708704, -6.413145462078729, -6.4193974021658535, -6.420028198057656, -6.397140024450819, -6.368447111407895, -6.334370134448919, -6.315541045488868, -6.307006550485679, -6.2918240091505995, -6.257989332027726, -6.227841821256566, -6.2150539033461065, -6.178442353395387, -6.14574723667004, -6.152790933449517, -6.142859449733626, -6.128059154306099, -6.111189914928695, -6.106107013643509, -6.0908853647608545, -6.047046580768666, -6.005456272325024, -5.975089839400411, -5.925203855560936, -5.9037147292996, -5.887455784981747, -5.8584467995755265, -5.844503788849716, -5.807221212160648, -5.777243257532861, -5.756736851048007, -5.709392762797486, -5.697783490588692, -5.698068676024961, -5.691713704921183, -5.66400885146795, -5.627537876161448, -5.603185199864221, -5.573766528058866, -5.551894313123237, -5.586781300612653, -5.541945112846908, -5.519466499320623, -5.483552927338986, -5.450121952065375, -5.446458546082542, -5.441432707022301, -5.4206919166131575, -5.410353152559493, -5.405354916569453, -5.399824078705348, -5.381524253978309, -5.3464596583866175, -5.354109225560658, -5.3407685214838185, -5.3138110119928665, -5.315943091227902, -5.287027110905253, -5.2427330439393875, -5.229471457445776, -5.227982069496862, -5.2021210229636505, -5.181106698115361, -5.182672726713714, -5.249781015311855, -5.294861987568179, -5.278789694280598, -5.26656313194652, -5.28353076982567, -5.2503877379744, -5.247253456183202, -5.254862162309286, -5.217999312988845, -5.182463418554816, -5.13915634417636, -5.101797954078596, -5.070848118870355, -5.042301742591523, -5.055324382604933, -5.027723819157941, -4.998571603083476, -4.967189624824594, -4.9322099395449905, -4.916531718588843, -4.910528293466666, -4.875380577533724, -4.834402343838057, -4.8094294050734225, -4.7770768240815675, -4.778094894063194, -4.755591806848735, -4.7272465799504895, -4.700650177736818, -4.702257125300788, -4.663214439027621, -4.629526982080008, -4.601420603573119, -4.614666502708079, -4.6125614606861065, -4.579546340187426, -4.571614666439881, -4.540760765729567, -4.51840144105611, -4.485371857462584, -4.458913869751103, -4.420845956709199, -4.399397648063961, -4.397273150843462, -4.363905886171823, -4.372932274187901, -4.368184468263209, -4.363277530523578, -4.3287059851789325, -4.3163477801218555, -4.281722347809052, -4.278271102552078, -4.2666460454152535, -4.264465712205054, -4.244412583011927, -4.218090543874355, -4.203195296905616, -4.17698464298421, -4.165712565507924, -4.129740718956894, -4.129941878126792, -4.114992090516834, -4.09012990248851, -4.0679970927379925, -4.043603826336644, -4.023321247917499, -4.013870783328057, -3.993242849586083, -3.974809180636937, -3.950281814565033, -3.944258924916637, -3.942573271419942, -3.9431756990877216, -3.9283266355516906, -3.9091800656731075, -3.904733792875116, -3.906773129745189, -3.8705637111134563, -3.868221595812635, -3.8443288877988766, -3.842542831100325, -3.813881246535047, -3.791494673590512, -3.781651010541429, -3.772894112062995, -3.743500332230549, -3.743020229163436, -3.7489250456575784, -3.7336830689549094, -3.7068705410285805, -3.6752036878840237, -3.6527127779711526, -3.6188438302894705, -3.596162027023082, -3.5958410471439772, -3.590409960390697, -3.5804851214819697, -3.5804794986732804, -3.5718461743139454, -3.5739844931734774, -3.5745278806687026, -3.571021083884216, -3.5727568460638435, -3.5582713689801966, -3.55758283189176, -3.554582537237479, -3.5515191921599714, -3.562182372221507, -3.558890965848756, -3.5594050927308727, -3.5626962772062503, -3.5643247986190905, -3.5624890124112767, -3.56890753899709, -3.5730360919122472, -3.5635535410203554, -3.5661414013420534, -3.559415825432311, -3.5622839062696134, -3.560150245736052, -3.5668089089103923, -3.565664231946113, -3.5652982541334577, -3.5695023981245737, -3.57666462839432, -3.5798379539640357, -3.5744703662892645, -3.5720383475422226, -3.569058457655264, -3.5607192078770327, -3.56302015404472, -3.5631441027412953, -3.564655009864751, -3.5680726575566495, -3.563977928484679, -3.5682371428604234, -3.5553220622547834, -3.552820015653321, -3.553165055652797, -3.553501747450184, -3.6229065754676952, -3.717250224318932, -3.8308298190118837, -3.895561360641827, -3.9980177211804557, -3.9922340856425693, -3.9827258979564637, -3.9726120074554805, -3.9753151829150655, -3.9744081466264367, -3.9700182438617238, -3.972443980068319, -3.9774137759391306, -3.9740386027683154, -3.9624794561665446, -3.954412768749631, -3.96322110368789, -3.9608451787349535, -3.960796333459614, -3.9604752872550124, -3.964410992567784, -3.9638670054729443, -3.964601455257103, -3.9646477832914173, -3.9662475669674815, -3.9724862231174716, -3.971054064034068, -3.9672592601749717, -3.9728335636839733, -3.979769938044458, -3.9877789828623116, -3.9922279195792743, -3.9948169125245747, -3.994335018375706, -4.001310263430944, -4.000455542793107, -4.002566917318526, -3.9984302975555863, -4.003892561829588, -3.9902317204026034, -3.981177557026573, -3.9871645539585865, -3.99636164201973, -3.995280481022809, -4.009033736168794, -4.002026539422152, -4.006175729466266, -4.001348087690829, -4.0037985221623105, -4.003598148764749, -4.00706989285482, -4.003907326320345, -4.001541817915259, -4.0014471642033955, -3.9928852472669765, -3.9958042618168017, -3.9943524071114673, -3.9990789483310296, -4.005589594513228, -4.009951616885376, -4.0066260363557005, -4.004364768235881, -3.9988572046952466, -3.9937861181385093, -3.99274191450587, -3.9955157120356097, -3.9893619254786175, -3.987398279440581, -3.9853386326698237, -3.9870635644671766, -3.9907643762427143, -3.991160556524238, -3.9887129279543836, -3.9882906319772715, -3.981122650943076, -3.973437853027626, -3.973236328963511, -3.971257842071025, -3.963931819802362, -3.968940548979369, -3.9666623921558157, -3.956560942180969, -3.9513375309617595, -3.9488555192065906, -3.9506826900673557, -3.9477734348835996, -3.9447023274860964, -3.9449219643475937, -3.9443677137298683, -3.947477345677397, -3.9481702669866214, -3.949669484137568, -3.952064525260377, -3.9588340232815744, -3.962213015412107, -3.965456205526466, -3.964933132799374, -3.9645941479476283, -3.9643536334832903, -3.9574990439612074, -3.959208945647034, -3.9509854520674104, -3.957837120107803, -3.9647526018853263, -3.9642988224939577, -3.9697317745061564, -3.9661853426821874, -3.968261115196877, -3.969570715783135, -3.9694979666823955, -3.974331057957087, -3.9667868716118027, -3.974588886155583, -3.971208070526126, -3.9751976743494657, -3.9749964054433056, -3.9754517618430283, -3.98541636081537, -3.989191954043375, -3.975350741352884, -3.976696515418111, -3.9679864006902297, -3.960419244222755, -3.9433314753044084, -3.9297692970542246, -3.9388682669015234, -3.938403483927601, -3.9537579508728773, -3.947699982175517, -3.9386795793301768, -3.918498244308385, -3.9365883793980827, -3.9417005722906073, -3.925481852871493, -3.9283005283523975, -3.9322847908874454, -3.9382629235788036, -3.9451916887719154, -3.9460521198794956, -3.9555451939769526, -3.9554619866031864, -3.9541424047044558, -3.9549189430772222, -3.957355088606598, -3.9344727709613063, -3.9384044921089867, -3.938574219477969, -3.929554150874414, -3.93462178026572, -3.9392689851571547, -3.9324996262022913, -3.9244117723336234, -3.92103732395183, -3.910531593219794, -3.913256363739742, -3.9196783272511757, -3.920208626984995, -3.917707859800345, -3.9056366807052516, -3.8895314518965223, -3.8872930794285447, -3.888626974325729, -3.89561684896222, -3.8955834909772644, -3.903186181737733, -3.8993278184296765, -3.909655103540158, -3.919831837535367, -3.921127116470277, -3.922189190303815, -3.9323330775938325, -3.93301668084726, -3.9397954424168713, -3.9407566524955446, -3.944696113078906, -3.946468216646588, -3.948173799046144, -3.9448985399790066, -3.9468066033190143, -3.9525579841895397, -3.9590402603767982, -3.9649215569597853, -3.967529529435693, -3.9693035283719276, -3.9732129108053984, -3.977148596375867, -3.9752676851281086, -3.9795030024354023, -3.9812424772828794, -3.984528822419164, -3.9854343170014768, -3.9910627530164744, -3.9949178260732103, -3.9954826010462865, -4.003009522476136, -4.004459746892136, -4.011990191142402, -4.020934221727561, -4.019424869483428, -4.019180644982729, -4.019788779017942, -4.0257716493576705, -4.033972776157101, -4.031133545350323, -4.03079122987459, -4.040906880856426, -4.042782632794009, -4.050156883327141, -4.050314737134725, -4.050372491201697, -4.049460441590246, -4.051115584031018, -4.049336029308151, -4.052204749787126, -4.046123648418427, -4.04253770619474, -4.034574837279567, -4.039063871960264, -4.03613109649518, -4.038245386349532, -4.0342214254513005, -4.038571496650392, -4.035447227340356, -4.036748240745292, -4.035063122538169, -4.044650411109631, -4.0466783757441265, -4.042864809408776, -4.042494084642017, -4.046801546055614, -4.052367298892969, -4.057389934039798, -4.058798788280988, -4.050489135435731, -4.050431825344777, -4.0505738134291756, -4.049527877596771, -4.04593573980092, -4.047832362219691, -4.045544757106005, -4.050888436712868, -4.055022041664517, -4.046933506646389, -4.056680441029624, -4.0565705843939295, -4.0549638509732855, -4.054918488696363, -4.050241655039169, -4.047519691630613, -4.046264687349796, -4.047284791502675, -4.040517440588242, -4.04156171872063, -4.04565470016669, -4.043808067799916, -4.04353463015967, -4.043953891950072, -4.053424257725866, -4.051396392624064, -4.04706183388111, -4.048087251076663, -4.049162078210443, -4.049332064731112, -4.044772889269946, -4.050765126242614, -4.051314930063671, -4.057299620621447, -4.065383406257973, -4.062674374541337, -4.058732704593505, -4.059055089105374, -4.0555611099951685, -4.054480476645035]
track_tries = [2, 0, 4, 2, 4, 4, 2, 2, 2, 1, 1, 1, 3, 0, 4, 0, 1, 1, 2, 0, 1, 2, 1, 3, 0, 1, 2, 3, 0, 2, 3, 3, 2, 1, 3, 0, 4, 0]
exploring = [98, 93, 70, 77, 96, 68, 116, 85, 108, 86, 111, 106, 71, 55, 59, 131, 89, 173, 54, 55, 143, 78, 226, 73, 67, 94, 98, 131, 144, 58, 43, 103, 163, 110, 80, 104, 188, 79, 105, 77, 85, 42, 91, 132, 152, 91, 101, 100, 115, 61, 87, 71, 79, 107, 98, 72, 88, 91, 54, 157, 46, 62, 155, 129, 85, 97, 76, 112, 47, 118, 114, 118, 125, 68, 142, 66, 73, 101, 231, 128, 232, 101, 64, 93, 104, 92, 264, 106, 116, 78, 141, 92, 59, 92, 62, 151, 68, 91, 99, 155, 102, 69, 68, 80, 135, 136, 72, 115, 46, 63, 70, 71, 127, 85, 188, 76, 91, 144, 105, 51, 61, 91, 99, 88, 106, 68, 150, 114, 113, 141, 79, 90, 75, 145, 54, 74, 115, 103, 62, 61, 114, 178, 40, 100, 58, 26, 137, 83, 49, 121, 126, 137, 87, 117, 50, 130, 101, 125, 156, 40, 45, 97, 45, 57, 68, 47, 87, 103, 45, 83, 68, 58, 43, 47, 90, 68, 112, 56, 42, 41, 22, 151, 59, 104, 120, 55, 70, 91, 66, 49, 53, 93, 82, 77, 77, 124, 93, 139, 91, 69, 129, 81, 96, 43, 41, 53, 66, 67, 29, 116, 110, 25, 116, 149, 75, 80, 63, 77, 91, 150, 80, 16, 54, 34, 30, 49, 66, 85, 92, 13, 22, 86, 13, 75, 64, 67, 86, 20, 24, 11, 29, 15, 33, 63, 78, 33, 24, 11, 102, 12, 49, 15, 110, 92, 52, 62, 66, 43, 65, 69, 37, 48, 36, 71, 48, 101, 47, 82, 55, 63, 114, 93, 95, 50, 48, 134, 15, 45, 11, 51, 81, 56, 9, 59, 47, 56, 71, 50, 60, 127, 216, 83, 107, 72, 104, 106, 79, 40, 8, 30, 54, 5, 53, 94, 84, 65, 46, 35, 27, 31, 33, 70, 47, 30, 13, 92, 19, 10, 25, 23, 71, 51, 65, 72, 10, 10, 11, 11, 87, 45, 18, 28, 12, 19, 7, 9, 10, 40, 41, 15, 30, 17, 11, 37, 40, 9, 20, 7, 9, 44, 88, 34, 57, 92, 48, 36, 25, 76, 39, 8, 11, 19, 6, 29, 52, 6, 34, 55, 7, 9, 27, 18, 10, 8, 78, 70, 3, 8, 3, 3, 1, 6, 58, 14, 8, 6, 4, 35, 47, 4, 5, 35, 8, 109, 19, 6, 41, 103, 6, 4, 6, 10, 37, 5, 50, 12, 13, 5, 4, 4, 3, 193, 5, 86, 92, 43, 24, 36, 6, 39, 17, 80, 57, 3, 18, 6, 5, 4, 28, 59, 3, 8, 4, 5, 60, 3, 19, 4, 31, 74, 33, 20, 11, 13, 63, 6, 48, 39, 75, 11, 38, 17, 10, 7, 39, 74, 103, 8, 5, 2, 10, 3, 63, 175, 94, 108, 39, 147, 164, 158, 77, 40, 77, 72, 105, 97, 58, 146, 151, 145, 58, 73, 127, 71, 89, 103, 122, 123, 159, 57, 71, 62, 76, 135, 74, 93, 138, 78, 80, 62, 120, 96, 124, 143, 61, 55, 111, 135, 12, 21, 23, 20, 26, 43, 98, 59, 130, 77, 83, 77, 69, 112, 54, 125, 57, 69, 60, 133, 116, 57, 142, 49, 53, 104, 54, 133, 125, 141, 77, 52, 147, 98, 88, 112, 85, 86, 124, 119, 52, 112, 78, 82, 79, 40, 118, 115, 77, 93, 101, 98, 59, 98, 121, 79, 74, 83, 119, 109, 53, 53, 124, 65, 59, 78, 101, 57, 75, 119, 96, 90, 56, 59, 88, 66, 122, 54, 113, 83, 37, 60, 83, 105, 153, 129, 134, 48, 84, 89, 89, 99, 121, 37, 67, 83, 46, 65, 85, 96, 95, 60, 152, 55, 79, 66, 60, 80, 73, 95, 77, 10, 99, 47, 14, 37, 38, 93, 71, 7, 14, 20, 9, 16, 32, 76, 72, 24, 11, 6, 9, 9, 49, 6, 32, 80, 68, 99, 79, 103, 27, 30, 14, 38, 8, 91, 14, 8, 97, 105, 25, 41, 83, 40, 71, 72, 15, 8, 8, 8, 9, 36, 45, 58, 96, 70, 135, 77, 66, 109, 132, 131, 135, 43, 75, 46, 79, 151, 77, 98, 89, 38, 87, 54, 76, 59, 178, 130, 38, 37, 39, 94, 66, 67, 96, 121, 100, 93, 88, 63, 56, 60, 90, 96, 131, 86, 77, 117, 51, 93, 57, 97, 66, 79, 57, 63, 64, 79, 72, 68, 51, 95, 108, 112, 91, 102, 109, 75, 61, 55, 95, 75, 70, 64, 93, 59, 91, 43, 53, 34, 58, 34, 9, 45, 65, 73, 78, 76, 53, 95, 61, 66, 40, 45, 72, 59, 142, 69, 55, 53, 72, 80, 91, 94, 29, 58, 8, 119, 70, 62, 77, 71, 146]
exploiting = [84, 92, 83, 75, 64, 56, 104, 96, 103, 71, 119, 126, 85, 39, 79, 126, 86, 172, 36, 65, 181, 100, 248, 73, 49, 126, 106, 130, 118, 76, 46, 101, 141, 132, 120, 108, 219, 80, 112, 92, 116, 45, 63, 153, 199, 98, 107, 112, 150, 61, 113, 88, 119, 147, 115, 87, 98, 120, 65, 187, 51, 75, 200, 165, 169, 132, 78, 153, 56, 163, 140, 146, 147, 99, 170, 67, 99, 163, 309, 153, 289, 125, 68, 145, 134, 134, 353, 131, 140, 123, 191, 171, 75, 166, 90, 260, 115, 111, 125, 223, 159, 95, 88, 127, 186, 191, 92, 148, 81, 88, 95, 113, 232, 138, 306, 104, 145, 248, 143, 77, 102, 147, 168, 141, 159, 95, 237, 165, 177, 209, 134, 159, 147, 263, 92, 120, 148, 216, 79, 79, 212, 283, 46, 138, 77, 49, 285, 181, 74, 246, 162, 290, 155, 227, 100, 260, 145, 202, 332, 84, 107, 162, 56, 105, 139, 51, 182, 205, 63, 173, 146, 125, 80, 108, 161, 137, 225, 78, 63, 81, 39, 305, 130, 231, 216, 130, 137, 176, 138, 82, 97, 167, 148, 133, 188, 276, 158, 272, 209, 118, 251, 156, 158, 68, 118, 118, 116, 116, 67, 233, 218, 31, 207, 311, 151, 139, 139, 167, 206, 377, 156, 44, 104, 48, 63, 110, 162, 212, 211, 30, 27, 216, 27, 173, 140, 130, 173, 30, 62, 37, 36, 22, 48, 179, 218, 58, 40, 34, 250, 39, 104, 39, 261, 175, 152, 177, 157, 92, 138, 177, 82, 118, 120, 162, 100, 211, 122, 187, 125, 163, 250, 210, 263, 141, 114, 287, 30, 120, 20, 121, 191, 84, 26, 117, 110, 155, 157, 127, 175, 361, 539, 234, 234, 189, 302, 227, 221, 131, 27, 102, 161, 28, 138, 238, 197, 198, 113, 98, 55, 58, 101, 229, 166, 62, 32, 269, 44, 35, 66, 73, 193, 149, 175, 247, 19, 33, 24, 26, 322, 98, 43, 70, 25, 64, 39, 30, 53, 104, 92, 59, 92, 53, 30, 120, 85, 29, 63, 30, 35, 118, 262, 84, 169, 314, 147, 133, 64, 201, 100, 26, 53, 37, 21, 77, 173, 40, 98, 193, 44, 40, 80, 66, 32, 22, 262, 245, 23, 19, 22, 22, 24, 28, 174, 40, 21, 23, 23, 96, 153, 23, 21, 71, 21, 435, 50, 20, 134, 374, 21, 22, 23, 27, 126, 22, 120, 18, 26, 23, 23, 22, 24, 758, 21, 274, 299, 167, 100, 130, 31, 124, 37, 276, 224, 23, 72, 22, 25, 21, 105, 152, 24, 23, 22, 22, 272, 24, 61, 23, 107, 258, 141, 100, 33, 80, 276, 23, 184, 122, 313, 28, 144, 63, 54, 26, 185, 284, 420, 23, 21, 24, 31, 23, 203, 602, 461, 413, 140, 603, 640, 565, 310, 173, 327, 314, 425, 385, 221, 646, 525, 673, 210, 354, 485, 306, 400, 409, 532, 423, 716, 226, 322, 241, 300, 618, 309, 402, 705, 356, 280, 241, 567, 406, 606, 740, 299, 349, 520, 714, 37, 60, 71, 47, 101, 152, 442, 248, 533, 363, 396, 358, 346, 573, 230, 560, 291, 272, 240, 710, 510, 248, 676, 217, 270, 421, 292, 582, 596, 590, 464, 302, 691, 512, 388, 491, 418, 359, 623, 543, 204, 584, 406, 482, 341, 190, 689, 578, 391, 572, 593, 540, 256, 600, 533, 481, 455, 427, 656, 523, 263, 267, 653, 405, 327, 363, 573, 343, 342, 570, 528, 527, 321, 325, 509, 355, 716, 333, 612, 453, 186, 359, 421, 589, 851, 806, 738, 301, 551, 537, 487, 589, 737, 186, 411, 479, 248, 386, 501, 513, 627, 296, 893, 394, 315, 415, 361, 507, 464, 535, 442, 65, 558, 273, 160, 228, 139, 441, 377, 37, 94, 76, 78, 50, 115, 502, 406, 116, 68, 65, 23, 39, 241, 37, 181, 571, 362, 593, 496, 772, 175, 173, 134, 200, 25, 688, 145, 49, 675, 734, 148, 192, 578, 392, 460, 521, 85, 60, 49, 44, 80, 200, 311, 366, 577, 505, 923, 541, 473, 708, 847, 960, 1065, 291, 630, 254, 598, 963, 482, 581, 607, 228, 625, 279, 618, 371, 1173, 832, 234, 284, 299, 754, 596, 490, 713, 816, 555, 649, 571, 417, 519, 386, 633, 592, 961, 746, 605, 892, 437, 720, 459, 742, 389, 500, 377, 465, 482, 726, 611, 416, 319, 596, 732, 870, 639, 767, 779, 545, 451, 345, 732, 650, 614, 589, 759, 532, 784, 255, 416, 274, 500, 215, 82, 568, 485, 507, 686, 683, 424, 648, 570, 548, 298, 423, 638, 509, 1316, 642, 376, 449, 597, 555, 835, 694, 199, 529, 102, 914, 547, 441, 685, 466, 1252]
window = np.ones(60)/60

#losses = np.convolve(losses, window, mode = 'valid')

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.title('Training Losses')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Plotting the rewards
plt.subplot(1, 2, 2)
plt.plot(avg_r, label='Rewards', color='r')
plt.title('Training Rewards')
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(track_tries)
plt.title('Number of Incorrect Guesses for Each Success')
plt.xlabel('Solution Number')
plt.ylabel('Number of Tries Used')
plt.show()

plt.figure()
plt.plot(exploring, label = "Exploration")
plt.plot(exploiting, label = "Exploitation")
plt.title('Exploration vs. Exploitation over Episodes')
plt.xlabel('Episode')
plt.ylabel('Actions Taken')
plt.legend()
plt.show()

window = np.ones(10)/10

smoothed_expr = np.convolve(exploring, window, mode = 'valid')
smoothed_expt = np.convolve(exploiting, window, mode = 'valid')
plt.figure()
plt.plot(smoothed_expr, label = "Exploration")
plt.plot(smoothed_expt, label = "Exploitation")
plt.title('Exploration vs. Exploitation over Episodes')
plt.xlabel('Episode')
plt.ylabel('Actions Taken')
plt.legend()
plt.show()
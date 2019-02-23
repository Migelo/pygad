'''
for creation of (good) colormaps see:
    * viscm (https://github.com/matplotlib/viscm)
    * http://colormap.org
'''
__all__ = ['cm_age', 'cm_k_g', 'cm_k_p',
           'cm_isolum', 'cm_my_viridis']

import matplotlib as mpl
import matplotlib.cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

def _create_new_listed_cm(name, data, bad='black'):
    cmap = ListedColormap(data, name=name)
    cmap.set_bad(bad)
    mpl.cm.register_cmap(name=name, cmap=cmap)
    return cmap

cm_age = LinearSegmentedColormap('age',
        {'red':   ((0.0,  0.45, 0.45),
                   (0.1,  0.80, 0.80),
                   (0.15, 0.90, 0.90),
                   (0.25, 1.00, 1.00),
                   (1.0,  1.00, 1.00)),
         'green': ((0.0,  0.50, 0.50),
                   (0.1,  0.82, 0.82),
                   (0.15, 0.85, 0.85),
                   (0.30, 0.60, 0.60),
                   (1.0,  0.25, 0.25)),
         'blue':  ((0.0,  0.95, 0.95),
                   (0.1,  0.90, 0.90),
                   (0.15, 0.85, 0.85),
                   (0.30, 0.15, 0.15),
                   (1.0,  0.05, 0.05))
        })
cm_age.set_bad('black')
mpl.cm.register_cmap(name='age', cmap=cm_age)

cm_k_g = LinearSegmentedColormap('BlackGreen',
        {'red':   ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (1.0, 0.3, 0.3)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'blue':  ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (1.0, 0.3, 0.3))
        })
cm_k_g.set_bad('black')
mpl.cm.register_cmap(name='BlackGreen', cmap=cm_k_g)

cm_k_p = LinearSegmentedColormap('BlackPurple',
        {'red':   ((0.0, 0.0, 0.0),
                   (1.0, 0.4, 0.4)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.1, 0.1)),
         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0))
        })
cm_k_p.set_bad('black')
mpl.cm.register_cmap(name='BlackPurple', cmap=cm_k_p)

cm_my_viridis = [[ 0.2357339 , 0.040361  , 0.32992874],
                 [ 0.23597094, 0.049147  , 0.33292911],
                 [ 0.23625984, 0.05706905, 0.33593124],
                 [ 0.23652753, 0.06443432, 0.33894114],
                 [ 0.23677196, 0.07136195, 0.34196013],
                 [ 0.23698389, 0.07794398, 0.3449902 ],
                 [ 0.23716406, 0.08423754, 0.34803261],
                 [ 0.23731975, 0.09027944, 0.351088  ],
                 [ 0.23743994, 0.09611554, 0.35415872],
                 [ 0.23753328, 0.10176465, 0.35724525],
                 [ 0.23759682, 0.10725192, 0.36034921],
                 [ 0.23762974, 0.1125962 , 0.36347199],
                 [ 0.23763264, 0.11781208, 0.36661485],
                 [ 0.23760703, 0.12291118, 0.36977886],
                 [ 0.23754749, 0.12790827, 0.37296603],
                 [ 0.23745543, 0.13281132, 0.37617747],
                 [ 0.23733608, 0.13762473, 0.37941357],
                 [ 0.23717774, 0.14236306, 0.38267736],
                 [ 0.23699179, 0.14702489, 0.38596813],
                 [ 0.23676475, 0.15162372, 0.38928929],
                 [ 0.23650939, 0.15615646, 0.39263955],
                 [ 0.23620972, 0.16063626, 0.39602294],
                 [ 0.23587529, 0.16506119, 0.3994385 ],
                 [ 0.23550356, 0.16943583, 0.4028875 ],
                 [ 0.23508313, 0.17376865, 0.40637325],
                 [ 0.23462224, 0.17805778, 0.40989456],
                 [ 0.23411801, 0.18230682, 0.41345263],
                 [ 0.2335723 , 0.1865342 , 0.41697488],
                 [ 0.23296722, 0.19075662, 0.42043312],
                 [ 0.23230042, 0.19497328, 0.42383639],
                 [ 0.23157107, 0.19918508, 0.42718307],
                 [ 0.23078065, 0.20339379, 0.43046265],
                 [ 0.22993001, 0.20759987, 0.43367137],
                 [ 0.22901708, 0.21180237, 0.43681748],
                 [ 0.22804263, 0.21600176, 0.43989719],
                 [ 0.22701093, 0.22020031, 0.44289146],
                 [ 0.22591847, 0.2243957 , 0.44581751],
                 [ 0.22476644, 0.22858794, 0.4486727 ],
                 [ 0.22356121, 0.23277923, 0.45143435],
                 [ 0.22229878, 0.2369668 , 0.45412297],
                 [ 0.2209821 , 0.24115085, 0.4567321 ],
                 [ 0.21961737, 0.24533231, 0.45924616],
                 [ 0.21820114, 0.24950873, 0.4616837 ],
                 [ 0.21674016, 0.25368072, 0.46403031],
                 [ 0.21523698, 0.25784745, 0.46628658],
                 [ 0.21369032, 0.26200721, 0.46846457],
                 [ 0.2121129 , 0.26616101, 0.47053901],
                 [ 0.21049946, 0.27030641, 0.47253335],
                 [ 0.20885761, 0.27444314, 0.47443859],
                 [ 0.20719376, 0.27857057, 0.47625033],
                 [ 0.20550497, 0.282687  , 0.47798546],
                 [ 0.20380728, 0.28679272, 0.47962052],
                 [ 0.2020948 , 0.29088588, 0.48117884],
                 [ 0.20037617, 0.29496601, 0.48265383],
                 [ 0.19865924, 0.2990325 , 0.48404186],
                 [ 0.19693997, 0.30308412, 0.48536091],
                 [ 0.19523607, 0.30712074, 0.48659103],
                 [ 0.19354141, 0.31114137, 0.48775276],
                 [ 0.19186181, 0.31514552, 0.4888464 ],
                 [ 0.19021061, 0.31913288, 0.48986142],
                 [ 0.18857939, 0.3231029 , 0.49081925],
                 [ 0.18698072, 0.32705532, 0.49171085],
                 [ 0.18541858, 0.33098993, 0.49253907],
                 [ 0.183888  , 0.33490661, 0.49331806],
                 [ 0.18240267, 0.33880523, 0.49403764],
                 [ 0.18096238, 0.34268582, 0.4947049 ],
                 [ 0.17956286, 0.34654862, 0.49533082],
                 [ 0.17821416, 0.35039368, 0.49590909],
                 [ 0.17692043, 0.3542211 , 0.49644086],
                 [ 0.17567365, 0.35803152, 0.49693863],
                 [ 0.17447558, 0.36182525, 0.49740463],
                 [ 0.17334336, 0.36560208, 0.49782596],
                 [ 0.17226203, 0.36936322, 0.49821892],
                 [ 0.17123139, 0.37310914, 0.49858633],
                 [ 0.17025029, 0.37684071, 0.49892904],
                 [ 0.169296  , 0.38056028, 0.4992619 ],
                 [ 0.16834639, 0.38427097, 0.4995917 ],
                 [ 0.1673984 , 0.38797359, 0.49991893],
                 [ 0.16645088, 0.39166873, 0.50024323],
                 [ 0.16550702, 0.39535663, 0.50056189],
                 [ 0.16456979, 0.39903754, 0.50087226],
                 [ 0.16362873, 0.40271281, 0.50117818],
                 [ 0.16268268, 0.40638294, 0.50147931],
                 [ 0.16172967, 0.41004855, 0.50177484],
                 [ 0.16077491, 0.41370964, 0.50206097],
                 [ 0.15981688, 0.41736668, 0.50233758],
                 [ 0.1588469 , 0.42102079, 0.50260669],
                 [ 0.15786305, 0.42467248, 0.50286752],
                 [ 0.15686374, 0.4283222 , 0.50311943],
                 [ 0.15585057, 0.43197014, 0.50336017],
                 [ 0.1548277 , 0.43561634, 0.50358634],
                 [ 0.15378408, 0.43926186, 0.50380142],
                 [ 0.15271701, 0.44290727, 0.50400407],
                 [ 0.15162465, 0.44655295, 0.50419346],
                 [ 0.15050543, 0.4501992 , 0.50436885],
                 [ 0.14936121, 0.4538462 , 0.50452725],
                 [ 0.14819468, 0.45749385, 0.50466715],
                 [ 0.14699459, 0.4611433 , 0.50478978],
                 [ 0.14575953, 0.46479474, 0.50489445],
                 [ 0.14448718, 0.46844854, 0.50497992],
                 [ 0.14317556, 0.47210497, 0.5050451 ],
                 [ 0.14182291, 0.47576427, 0.50508903],
                 [ 0.14043093, 0.47942646, 0.50510915],
                 [ 0.13900174, 0.48309147, 0.50510383],
                 [ 0.13752492, 0.48676016, 0.5050737 ],
                 [ 0.13599942, 0.49043258, 0.50501809],
                 [ 0.13442229, 0.49410905, 0.50493534],
                 [ 0.13279254, 0.49778961, 0.50482477],
                 [ 0.13110735, 0.50147451, 0.5046848 ],
                 [ 0.12936541, 0.5051638 , 0.50451458],
                 [ 0.12756443, 0.50885763, 0.50431279],
                 [ 0.12570232, 0.5125561 , 0.5040782 ],
                 [ 0.123787  , 0.51625871, 0.50380788],
                 [ 0.12180582, 0.51996621, 0.50350222],
                 [ 0.11975884, 0.52367841, 0.50316113],
                 [ 0.11764186, 0.5273956 , 0.50278244],
                 [ 0.11545391, 0.53111766, 0.50236557],
                 [ 0.11319279, 0.53484461, 0.50190935],
                 [ 0.11085542, 0.53857652, 0.50141233],
                 [ 0.10844071, 0.54231325, 0.50087392],
                 [ 0.10594564, 0.54605483, 0.5002928 ],
                 [ 0.10336666, 0.54980132, 0.49966747],
                 [ 0.10070321, 0.55355244, 0.49899779],
                 [ 0.09795066, 0.55730831, 0.49828198],
                 [ 0.09510597, 0.56106883, 0.49751908],
                 [ 0.09216619, 0.56483388, 0.49670825],
                 [ 0.08912835, 0.56860331, 0.4958488 ],
                 [ 0.08598629, 0.57237719, 0.49493895],
                 [ 0.0827363 , 0.57615532, 0.49397811],
                 [ 0.07937427, 0.57993751, 0.49296573],
                 [ 0.07589331, 0.58372373, 0.4919005 ],
                 [ 0.07227909, 0.58751435, 0.49077896],
                 [ 0.06857284, 0.59130674, 0.48960664],
                 [ 0.06476845, 0.59510138, 0.48837678],
                 [ 0.06087449, 0.59889765, 0.48708886],
                 [ 0.0569011 , 0.60269502, 0.48574213],
                 [ 0.05285983, 0.60649299, 0.48433544],
                 [ 0.04876207, 0.61029128, 0.48286656],
                 [ 0.04463329, 0.61408907, 0.48133606],
                 [ 0.040513  , 0.61788551, 0.47974299],
                 [ 0.0364884 , 0.62168015, 0.47808547],
                 [ 0.03278111, 0.62547261, 0.47636339],
                 [ 0.02942294, 0.62926238, 0.47457556],
                 [ 0.0264439 , 0.63304902, 0.47272024],
                 [ 0.02388344, 0.63683168, 0.47079833],
                 [ 0.02176808, 0.64061018, 0.46880637],
                 [ 0.02014151, 0.64438354, 0.46674614],
                 [ 0.01903432, 0.64815145, 0.46461475],
                 [ 0.01848859, 0.6519131 , 0.46241296],
                 [ 0.01853935, 0.65566804, 0.46013864],
                 [ 0.01924498, 0.6594151 , 0.45779274],
                 [ 0.02069884, 0.66315239, 0.4553752 ],
                 [ 0.02288076, 0.6668808 , 0.45288451],
                 [ 0.02582974, 0.67059989, 0.45031804],
                 [ 0.02959321, 0.67430882, 0.44767699],
                 [ 0.03421315, 0.6780071 , 0.44495907],
                 [ 0.03973665, 0.68169399, 0.44216434],
                 [ 0.04597372, 0.68536665, 0.43929595],
                 [ 0.05257531, 0.6890257 , 0.43634958],
                 [ 0.05946148, 0.69267117, 0.43332547],
                 [ 0.066588  , 0.69630246, 0.43022215],
                 [ 0.07392045, 0.69991895, 0.42703808],
                 [ 0.08149411, 0.70351741, 0.42377776],
                 [ 0.08924901, 0.7070983 , 0.42043895],
                 [ 0.09713699, 0.71066216, 0.41701865],
                 [ 0.10514587, 0.71420836, 0.41351563],
                 [ 0.11329847, 0.71773443, 0.40993344],
                 [ 0.12159726, 0.72123883, 0.40627276],
                 [ 0.12998595, 0.72472335, 0.40252905],
                 [ 0.13846005, 0.72818735, 0.39870132],
                 [ 0.14706117, 0.73162719, 0.39479426],
                 [ 0.1557622 , 0.73504333, 0.39080613],
                 [ 0.16453361, 0.73843687, 0.38673282],
                 [ 0.17338644, 0.74180615, 0.38257617],
                 [ 0.18235605, 0.7451474 , 0.37834169],
                 [ 0.19138601, 0.74846408, 0.37402108],
                 [ 0.20047584, 0.75175552, 0.36961404],
                 [ 0.20968235, 0.7550158 , 0.36513094],
                 [ 0.2189416 , 0.75824969, 0.36056065],
                 [ 0.22825341, 0.7614566 , 0.35590252],
                 [ 0.23767073, 0.76463038, 0.35116831],
                 [ 0.24713432, 0.76777618, 0.34634549],
                 [ 0.2566513 , 0.77089266, 0.34143501],
                 [ 0.26625583, 0.77397517, 0.33644621],
                 [ 0.27590056, 0.77702836, 0.33136673],
                 [ 0.28560062, 0.78005017, 0.32619309],
                 [ 0.29527125, 0.78304998, 0.32092147],
                 [ 0.30490484, 0.78603005, 0.31552543],
                 [ 0.31452614, 0.78898658, 0.31003182],
                 [ 0.32413593, 0.79192075, 0.30440897],
                 [ 0.33375144, 0.79482962, 0.29867531],
                 [ 0.34337645, 0.79771312, 0.29281491],
                 [ 0.35302603, 0.80056854, 0.28683753],
                 [ 0.36270172, 0.80339596, 0.28072422],
                 [ 0.37242043, 0.8061916 , 0.27449891],
                 [ 0.38218001, 0.80895622, 0.26813076],
                 [ 0.39199639, 0.81168604, 0.26163951],
                 [ 0.40187207, 0.81438002, 0.25502043],
                 [ 0.41181242, 0.81703678, 0.24826133],
                 [ 0.42182577, 0.8196537 , 0.24137114],
                 [ 0.43192059, 0.82222776, 0.23436765],
                 [ 0.44209906, 0.82475785, 0.227236  ],
                 [ 0.45236651, 0.82724156, 0.21998491],
                 [ 0.46272735, 0.8296765 , 0.21262609],
                 [ 0.47318619, 0.83206015, 0.20517323],
                 [ 0.48374677, 0.83438999, 0.19764413],
                 [ 0.49440931, 0.836664  , 0.19006547],
                 [ 0.50517489, 0.83887997, 0.18246742],
                 [ 0.5160429 , 0.84103595, 0.17488861],
                 [ 0.52700873, 0.84313069, 0.16738091],
                 [ 0.53806738, 0.84516293, 0.16001645],
                 [ 0.54921177, 0.84713149, 0.15290466],
                 [ 0.56042778, 0.84903805, 0.14611126],
                 [ 0.57170126, 0.85088383, 0.13975507],
                 [ 0.58301593, 0.85266987, 0.13403525],
                 [ 0.59434632, 0.85440197, 0.12903365],
                 [ 0.60567161, 0.85608327, 0.12496942],
                 [ 0.61696289, 0.85772109, 0.12195615],
                 [ 0.62819437, 0.85932172, 0.12015125],
                 [ 0.63933758, 0.86089914, 0.11923786],
                 [ 0.65037312, 0.86246797, 0.11868442],
                 [ 0.66130419, 0.86402895, 0.11851183],
                 [ 0.67213351, 0.86558286, 0.11874032],
                 [ 0.68286427, 0.86713037, 0.11937892],
                 [ 0.69349709, 0.86867259, 0.12045049],
                 [ 0.7040354 , 0.87021022, 0.12194131],
                 [ 0.714478  , 0.87174457, 0.12388399],
                 [ 0.72482848, 0.87327638, 0.12624518],
                 [ 0.73508562, 0.87480705, 0.12904008],
                 [ 0.74525071, 0.87633764, 0.13225024],
                 [ 0.75532489, 0.87786925, 0.13585405],
                 [ 0.76530601, 0.87940347, 0.13986155],
                 [ 0.77519664, 0.8809412 , 0.14422839],
                 [ 0.78499572, 0.88248393, 0.14894671],
                 [ 0.79470261, 0.88403307, 0.1540029 ],
                 [ 0.80431884, 0.88558967, 0.15936101],
                 [ 0.8138427 , 0.88715534, 0.16501764],
                 [ 0.82327494, 0.88873124, 0.17094551],
                 [ 0.83261654, 0.89031849, 0.17711751],
                 [ 0.84186515, 0.89191879, 0.18353775],
                 [ 0.85102324, 0.89353295, 0.19016809],
                 [ 0.86009047, 0.89516226, 0.1969985 ],
                 [ 0.86906628, 0.89680803, 0.20402201],
                 [ 0.87796424, 0.89846667, 0.21121168],
                 [ 0.88683371, 0.90012606, 0.21832563],
                 [ 0.89568603, 0.90178517, 0.22526144],
                 [ 0.90452303, 0.90344375, 0.23203435],
                 [ 0.91334662, 0.9051015 , 0.2386577 ],
                 [ 0.92216121, 0.90675704, 0.24514467],
                 [ 0.93095948, 0.9084138 , 0.25150052],
                 [ 0.9397535 , 0.9100672 , 0.25774026],
                 [ 0.94854287, 0.91171773, 0.26387089],
                 [ 0.95733013, 0.91336464, 0.26990021],
                 [ 0.9661211 , 0.91500573, 0.27583689],
                 [ 0.97492879, 0.91663555, 0.28169216],
                 [ 0.98384554, 0.91821347, 0.28751257]]
cm_my_viridis
cm_my_viridis = _create_new_listed_cm('my_viridis', cm_my_viridis)
try:
    viridis = mpl.get_cmap('viridis')
    del viridis
except:
    mpl.cm.register_cmap(name='viridis', cmap=cm_my_viridis)

cm_isolum = [[ 0.60650245, 0.52403835, 0.96984564],
             [ 0.60010679, 0.52742294, 0.96805436],
             [ 0.59387394, 0.53063655, 0.96640654],
             [ 0.58763435, 0.53377172, 0.96485455],
             [ 0.58138359, 0.53683218, 0.96339827],
             [ 0.5751024 , 0.53982845, 0.96203442],
             [ 0.56878319, 0.54276472, 0.96076286],
             [ 0.56241238, 0.54564752, 0.9595825 ],
             [ 0.55599185, 0.54847607, 0.95849524],
             [ 0.5495054 , 0.55125738, 0.95749978],
             [ 0.54295525, 0.55399018, 0.95659783],
             [ 0.5363144 , 0.55668511, 0.9557871 ],
             [ 0.52958658, 0.55933969, 0.95506954],
             [ 0.52276332, 0.56195635, 0.9544453 ],
             [ 0.51583055, 0.56453931, 0.95391406],
             [ 0.5087827 , 0.56708925, 0.95347622],
             [ 0.50160171, 0.56961106, 0.95313133],
             [ 0.49426942, 0.57210905, 0.95287928],
             [ 0.48678389, 0.57458162, 0.95272047],
             [ 0.47913764, 0.57702899, 0.95265461],
             [ 0.47128776, 0.57946204, 0.95268137],
             [ 0.46334055, 0.58188277, 0.95259325],
             [ 0.45528325, 0.58431834, 0.95225049],
             [ 0.44713513, 0.58676618, 0.95162449],
             [ 0.43888827, 0.58922186, 0.95074251],
             [ 0.43057218, 0.59168178, 0.94956912],
             [ 0.42219648, 0.59414113, 0.94810982],
             [ 0.41376477, 0.59659507, 0.94638098],
             [ 0.40531401, 0.59903898, 0.94435102],
             [ 0.39685167, 0.60146751, 0.94203927],
             [ 0.38838601, 0.60387567, 0.93946375],
             [ 0.37995853, 0.6062584 , 0.93659997],
             [ 0.37157655, 0.60861085, 0.9334732 ],
             [ 0.36324709, 0.61092869, 0.93010738],
             [ 0.35501802, 0.61320716, 0.92647977],
             [ 0.34688829, 0.61544261, 0.92262698],
             [ 0.33886253, 0.61763212, 0.91857456],
             [ 0.3310047 , 0.61977127, 0.91428951],
             [ 0.3232811 , 0.62185904, 0.90983615],
             [ 0.31572933, 0.62389278, 0.90520606],
             [ 0.30836596, 0.6258705 , 0.90041381],
             [ 0.30117745, 0.62779248, 0.89549297],
             [ 0.29421858, 0.62965605, 0.89042346],
             [ 0.28744029, 0.63146335, 0.88526467],
             [ 0.28091706, 0.63321195, 0.87997766],
             [ 0.27458387, 0.63490484, 0.87462999],
             [ 0.26851463, 0.63654014, 0.86918255],
             [ 0.26264851, 0.63812164, 0.86368973],
             [ 0.25703411, 0.63964848, 0.85813117],
             [ 0.25164762, 0.6411228 , 0.85253431],
             [ 0.2464778 , 0.64254697, 0.84691123],
             [ 0.2415688 , 0.64392052, 0.84124352],
             [ 0.23685994, 0.64524781, 0.83557128],
             [ 0.23237387, 0.64652951, 0.82988507],
             [ 0.22812303, 0.64776667, 0.82418205],
             [ 0.22406009, 0.64896314, 0.81848813],
             [ 0.2201814 , 0.65012081, 0.81280427],
             [ 0.21653571, 0.6512386 , 0.80711085],
             [ 0.21305602, 0.6523216 , 0.80143544],
             [ 0.20973403, 0.65337172, 0.79577949],
             [ 0.20656091, 0.65439083, 0.79014434],
             [ 0.20356629, 0.65537861, 0.78451737],
             [ 0.20070732, 0.65633869, 0.77891165],
             [ 0.19796386, 0.65727356, 0.77332922],
             [ 0.19526047, 0.65818795, 0.76779513],
             [ 0.19251621, 0.6590903 , 0.76230412],
             [ 0.18973006, 0.65998133, 0.75685114],
             [ 0.18692381, 0.66085956, 0.75143255],
             [ 0.18406746, 0.66172793, 0.74604606],
             [ 0.18115725, 0.66258721, 0.74068737],
             [ 0.17819039, 0.66343787, 0.7353536 ],
             [ 0.17516409, 0.66428039, 0.73004145],
             [ 0.17211491, 0.66511219, 0.72474668],
             [ 0.16900788, 0.66593638, 0.71946603],
             [ 0.16583825, 0.66675345, 0.71419698],
             [ 0.16260444, 0.66756372, 0.70893573],
             [ 0.15932659, 0.66836597, 0.70367818],
             [ 0.15601791, 0.66915943, 0.69842142],
             [ 0.15264811, 0.66994648, 0.69316234],
             [ 0.14921831, 0.67072713, 0.68789776],
             [ 0.14575362, 0.6714999 , 0.68262386],
             [ 0.142278  , 0.67226346, 0.6773373 ],
             [ 0.13875529, 0.67302038, 0.67203523],
             [ 0.13519033, 0.67377049, 0.66671449],
             [ 0.13164285, 0.67451036, 0.66137208],
             [ 0.12809793, 0.67524129, 0.65600476],
             [ 0.12453989, 0.67596447, 0.65060964],
             [ 0.12100293, 0.67667831, 0.64518363],
             [ 0.11755124, 0.67737972, 0.63972423],
             [ 0.11413333, 0.67807207, 0.63422828],
             [ 0.11077148, 0.67875464, 0.62869289],
             [ 0.10758966, 0.67942177, 0.62311668],
             [ 0.10451182, 0.68007809, 0.617496  ],
             [ 0.10156479, 0.68072297, 0.61182832],
             [ 0.09890422, 0.68134995, 0.60611288],
             [ 0.09644311, 0.68196406, 0.6003458 ],
             [ 0.09423019, 0.68256394, 0.59452516],
             [ 0.09241304, 0.68314401, 0.58865046],
             [ 0.09090731, 0.68370891, 0.58271786],
             [ 0.08979711, 0.68425574, 0.57672637],
             [ 0.08916966, 0.68478149, 0.570675  ],
             [ 0.08895487, 0.68528954, 0.56456032],
             [ 0.08927723, 0.68577482, 0.55838267],
             [ 0.0901093 , 0.68623835, 0.55213971],
             [ 0.09141383, 0.68668123, 0.54582914],
             [ 0.09333275, 0.68709652, 0.53945282],
             [ 0.09571184, 0.68748968, 0.53300613],
             [ 0.09861631, 0.68785632, 0.52648992],
             [ 0.10202003, 0.68819575, 0.51990309],
             [ 0.10583384, 0.68851013, 0.51324265],
             [ 0.11015919, 0.68879221, 0.5065117 ],
             [ 0.11483095, 0.68904777, 0.49970515],
             [ 0.11989985, 0.68927172, 0.49282497],
             [ 0.12529471, 0.68946525, 0.48587008],
             [ 0.13096122, 0.6896291 , 0.47883776],
             [ 0.13694104, 0.68975802, 0.47173444],
             [ 0.14315319, 0.68985531, 0.46454474],
             [ 0.149669  , 0.68991264, 0.45728895],
             [ 0.15639082, 0.68993501, 0.44994595],
             [ 0.16339114, 0.68991358, 0.44253609],
             [ 0.17058435, 0.68985262, 0.43504347],
             [ 0.1780171 , 0.68974521, 0.42748104],
             [ 0.18564245, 0.68959245, 0.41984384],
             [ 0.1934586 , 0.68939185, 0.41213131],
             [ 0.20147632, 0.68913888, 0.40435716],
             [ 0.20965052, 0.68883551, 0.39650833],
             [ 0.21800031, 0.68847657, 0.38859787],
             [ 0.22651155, 0.68806041, 0.3806296 ],
             [ 0.23515833, 0.68758722, 0.37259872],
             [ 0.24393994, 0.68705417, 0.36451225],
             [ 0.25286693, 0.68645659, 0.35638591],
             [ 0.26190638, 0.68579602, 0.34821197],
             [ 0.2710511 , 0.6850706 , 0.33999683],
             [ 0.28029324, 0.68427861, 0.33174739],
             [ 0.28963038, 0.68341748, 0.32347596],
             [ 0.29905276, 0.68248585, 0.31519094],
             [ 0.30854226, 0.68148406, 0.30689489],
             [ 0.31808876, 0.68041129, 0.29859694],
             [ 0.32768153, 0.67926698, 0.29030674],
             [ 0.3373103 , 0.67805077, 0.28203357],
             [ 0.34696264, 0.67676282, 0.27378864],
             [ 0.35662648, 0.67540355, 0.26558289],
             [ 0.36629029, 0.67397359, 0.25742683],
             [ 0.37594121, 0.6724741 , 0.24933223],
             [ 0.38556685, 0.6709065 , 0.24131052],
             [ 0.39515529, 0.66927246, 0.23337283],
             [ 0.4046942 , 0.66757409, 0.22553075],
             [ 0.41417211, 0.66581368, 0.21779526],
             [ 0.42357796, 0.66399379, 0.2101771 ],
             [ 0.43289681, 0.6621181 , 0.20269084],
             [ 0.44215055, 0.66017999, 0.19541708],
             [ 0.45132403, 0.65818324, 0.1883802 ],
             [ 0.46039727, 0.65613409, 0.181579  ],
             [ 0.46935355, 0.65403853, 0.17501054],
             [ 0.47818106, 0.65190178, 0.16866886],
             [ 0.48687128, 0.64972853, 0.1625462 ],
             [ 0.49542339, 0.64752135, 0.15664263],
             [ 0.50383877, 0.64528208, 0.15096328],
             [ 0.51210453, 0.64301754, 0.14547872],
             [ 0.52022235, 0.64072995, 0.14018262],
             [ 0.52820875, 0.6384162 , 0.13509832],
             [ 0.53604425, 0.63608566, 0.13018014],
             [ 0.54374917, 0.63373391, 0.12544809],
             [ 0.5513153 , 0.63136623, 0.12088261],
             [ 0.55875218, 0.62898175, 0.11648183],
             [ 0.56606366, 0.62658142, 0.1122377 ],
             [ 0.57325194, 0.62416647, 0.10815143],
             [ 0.58031814, 0.6217389 , 0.10420166],
             [ 0.58727842, 0.61929428, 0.10041376],
             [ 0.59412137, 0.61683928, 0.09674429],
             [ 0.60086455, 0.61436848, 0.0932264 ],
             [ 0.60750616, 0.61188445, 0.08983903],
             [ 0.61404511, 0.60938941, 0.08656606],
             [ 0.62049607, 0.60687857, 0.08343566],
             [ 0.62685824, 0.60435375, 0.08043269],
             [ 0.63313133, 0.60181657, 0.07754231],
             [ 0.63931995, 0.59926636, 0.07476491],
             [ 0.64543386, 0.59669957, 0.07213411],
             [ 0.65147222, 0.59411802, 0.0696208 ],
             [ 0.65743659, 0.59152202, 0.0672224 ],
             [ 0.66332783, 0.58891213, 0.06494158],
             [ 0.6691519 , 0.58628655, 0.06277911],
             [ 0.67491408, 0.58364339, 0.0607514 ],
             [ 0.68061512, 0.58098306, 0.05885639],
             [ 0.68625642, 0.5783057 , 0.05708456],
             [ 0.69183872, 0.57561159, 0.05543868],
             [ 0.69736567, 0.57289951, 0.05392051],
             [ 0.70283912, 0.57016909, 0.0525322 ],
             [ 0.70826005, 0.5674203 , 0.05127581],
             [ 0.71363296, 0.56465129, 0.0501523 ],
             [ 0.71895599, 0.56186341, 0.04916362],
             [ 0.72423349, 0.55905472, 0.04830993],
             [ 0.72946712, 0.55622466, 0.04759147],
             [ 0.73465689, 0.55337347, 0.04700814],
             [ 0.73980452, 0.55050047, 0.04655887],
             [ 0.74491219, 0.54760463, 0.04624201],
             [ 0.74999521, 0.54467726, 0.04605481],
             [ 0.75504512, 0.54172379, 0.045941  ],
             [ 0.76006364, 0.53874323, 0.04589581],
             [ 0.76505032, 0.53573566, 0.04592473],
             [ 0.7700044 , 0.53270142, 0.04603279],
             [ 0.7749288 , 0.52963851, 0.0462247 ],
             [ 0.77982319, 0.52654691, 0.04650474],
             [ 0.78468638, 0.52342714, 0.0468783 ],
             [ 0.78951754, 0.52027943, 0.04735376],
             [ 0.79431976, 0.51710161, 0.04792879],
             [ 0.79909179, 0.51389422, 0.04860552],
             [ 0.80383727, 0.51065447, 0.04938637],
             [ 0.80855279, 0.50738432, 0.05027217],
             [ 0.81324302, 0.50408015, 0.05126478],
             [ 0.81790456, 0.50074389, 0.05236409],
             [ 0.82254078, 0.49737269, 0.05357107],
             [ 0.82714439, 0.49397118, 0.05489809],
             [ 0.83171752, 0.4905375 , 0.05634049],
             [ 0.83626574, 0.48706709, 0.05789116],
             [ 0.84078888, 0.48355942, 0.05954936],
             [ 0.8452868 , 0.48001394, 0.06131475],
             [ 0.84974559, 0.47644054, 0.06321244],
             [ 0.85418025, 0.47282727, 0.06521678],
             [ 0.85859085, 0.46917323, 0.06732696],
             [ 0.86296758, 0.46548536, 0.06956166],
             [ 0.86731309, 0.46176094, 0.07191483],
             [ 0.8716352 , 0.45799274, 0.07437386],
             [ 0.87591711, 0.45419375, 0.07696884],
             [ 0.88017158, 0.45035261, 0.07967717],
             [ 0.88439078, 0.44647507, 0.08251181],
             [ 0.88857297, 0.44256184, 0.08547656],
             [ 0.89272142, 0.43860903, 0.08857196],
             [ 0.89682858, 0.43462268, 0.09180503],
             [ 0.90089354, 0.43060271, 0.09518329],
             [ 0.90492052, 0.42654441, 0.09869944],
             [ 0.90888988, 0.42246536, 0.10238868],
             [ 0.91281307, 0.41835425, 0.10623356],
             [ 0.91668406, 0.41421607, 0.11024722],
             [ 0.92048985, 0.4100632 , 0.11445107],
             [ 0.92422616, 0.40589984, 0.11885441],
             [ 0.92788913, 0.40172998, 0.12346589],
             [ 0.93146977, 0.39756312, 0.12830309],
             [ 0.9349573 , 0.39341129, 0.13338407],
             [ 0.93833953, 0.38928875, 0.13872813],
             [ 0.94160271, 0.38521228, 0.1443559 ],
             [ 0.94472502, 0.38120866, 0.15029755],
             [ 0.94774852, 0.37723892, 0.15639249],
             [ 0.95078967, 0.37317981, 0.16231654],
             [ 0.95384454, 0.36903314, 0.16807788],
             [ 0.95691932, 0.3647867 , 0.17370162],
             [ 0.96000839, 0.36044418, 0.17918904],
             [ 0.9631164 , 0.35599466, 0.18455866],
             [ 0.96624499, 0.35143085, 0.18982184],
             [ 0.96939282, 0.34674931, 0.19498393],
             [ 0.97256299, 0.34193953, 0.20005672],
             [ 0.97575567, 0.33699466, 0.20504634],
             [ 0.97897088, 0.33190747, 0.2099581 ],
             [ 0.98221836, 0.32665413, 0.21481128],
             [ 0.98549912, 0.32122316, 0.21961088],
             [ 0.98888478, 0.31548059, 0.22446175]]
cm_isolum = _create_new_listed_cm('isolum', cm_isolum)

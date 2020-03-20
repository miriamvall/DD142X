WAVE_f = A sin (2pi f) where f is sampled from 
    N(10,   sigma     ) for f in [12, 30 )
    N(20, 3 sigma     ) for f in [ 4,  8 )
    N( 3,   sigma / 3 ) for f in [35, 100)

CHANNEL_n = SUM(f = [4, 8), [12, 30), [35, 100)) WAVE_f
Data is NORMALIZED vector = (vector - vector.mean()) / vector.std()

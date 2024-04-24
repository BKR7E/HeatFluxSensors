Three different files doing different things: 

    1) NusseltFunofAngleRen.py) Plots Nusselt # as a function of Angle and Reynolds #; this is the same code used in dataBinner.py; it also generates an HTML file that can be opened using a browser

    2) VelCalcfromHtcAng.py) Calculates velocity[m/s] from angle[deg] and heat transfer coefficient[W/m^2-K] from data using a FluxTeq PHFS-01 heat flux sensor. Populated with data from AngleHtcVel.csv

    3) HtcCalcfromAngVel.py) Calculates heat transfer coefficient[W/m^2-K] from angle[deg] and velocity[m/s] from data using a FluxTeq PHFS-01 heat flux sensor. Populated with data from AngleVelHtc.csv

Coded by Eric Alar, UW-Madison (4/24/24) with help from ChatGPT 3.5

x = Angle, y = RE, z = Nusselt #

![image](https://github.com/BKR7E/DataBinner/assets/124415162/881d5bfd-58db-4c74-abf2-0e8d1cd8bcce)
![image](https://github.com/BKR7E/DataBinner/assets/124415162/507044e0-dada-4c75-afae-79be6194a247)
![image](https://github.com/BKR7E/DataBinner/assets/124415162/d9e0f37f-2aca-4f91-8183-a1bc44ba9fb7)
![image](https://github.com/BKR7E/DataBinner/assets/124415162/5a024ecb-80f8-4636-9bac-67b5597e405a)
![image](https://github.com/BKR7E/DataBinner/assets/124415162/ee036427-5eb9-4446-b068-bb1cd4c90248)

from model_all import *

"""
6th floor
    operational therapists: 20
    robotic therapists: 6
    cleaner: 1

7th floor
    nurses: 19
    transfers: 2
    patients: 8 * 4(rooms)
    caregivers: 8 * 4(rooms)
    cleaners: 2

8th floor
    nurses: 17
    transfers: 6
    patients: 8 * 6(rooms)
    caregivers: 8 * 6(rooms)
    cleaners: 2

9th floor
    transfers: 2
    operational therapists: 3
    physical therapists: 26
    cleaner: 1

10th floor
    nurses: 12
    transfers: 2
    patients: 8 * 2(rooms) + 2(patients in room number 43)
    caregivers: 8 * 2(rooms) + 2(caregivers in room number 43)
    cleaners: 2
    
    
classifier: abcd
- a: occupation
- b: floor (1 for 10th floor)
- cd: id number for each floor, occupation
"""

def place_agent(sev_model:EpidemicsModel):
    ## 6th floor
    # operational therapists
    for i in range(20):
        classifier = "O6" + str(i)
        sev_model.add_agent(classifier=classifier,
                            occ=4,
                            mr=13,
                            stat = 0)
    
    # robotic therapists
    for i in range(6):
        classifier = "R6" + str(i)
        sev_model.add_agent(classifier=classifier,
                            occ=5,
                            mr=14,
                            stat = 0)
    
    # cleaner
    classifier = "W60"
    sev_model.add_agent(classifier=classifier,
                        occ=7,
                        mr=10,
                        stat = 0)
    
    
    ## 7th floor
    # nurses
    for i in range(19):
        classifier = "N7" + str(i)
        sev_model.add_agent(classifier=classifier,
                            occ=0,
                            mr=21,
                            stat=0)

    # transfers
    for i in range(2):
        classifier = "T7" + str(i)
        sev_model.add_agent(classifier=classifier,
                            occ=1,
                            mr=20,
                            stat=0)

    # patients
    for i in range(4):
        for j in range(8):
            classifier = "P7" + str(8*i + j)
            sev_model.add_agent(classifier=classifier,
                                occ=2,
                                mr=i+22,
                                stat=0)

    # caregivers
    for i in range(3):
        for j in range(8):
            classifier = "C7" + str(8*i + j)
            sev_model.add_agent(classifier=classifier,
                                occ=3,
                                mr=i+22,
                                stat=0)

    for j in range(7):
        classifier = "C7" + str(24 + j)
        sev_model.add_agent(classifier=classifier,
                            occ=3,
                            mr=25,
                            stat=0)
        
    # caregiver (Infectious)
    sev_model.add_agent(classifier="C731",
                        occ=3,
                        mr=25,
                        stat=2)
    
    # cleaners
    for i in range(2):
        classifier = "W7" + str(i)
        sev_model.add_agent(classifier=classifier,
                            occ=7,
                            mr=20,
                            stat=0)
    
    
    ## 8th floor
    # nurses
    for i in range(17):
        classifier = "N8" + str(i)
        sev_model.add_agent(classifier=classifier,
                            occ=0,
                            mr=31,
                            stat=0)

    # transfers
    for i in range(6):
        classifier = "T8" + str(i)
        sev_model.add_agent(classifier=classifier,
                            occ=1,
                            mr=30,
                            stat=0)

    # patients
    for i in range(6):
        for j in range(8):
            classifier = "P8" + str(8*i + j)
            sev_model.add_agent(classifier=classifier,
                                occ=2,
                                mr=i+32,
                                stat=0)

    # caregivers
    for i in range(6):
        for j in range(8):
            classifier = "C8" + str(8*i + j)
            sev_model.add_agent(classifier=classifier,
                                occ=3,
                                mr=i+32,
                                stat=0)
            
    # cleaners
    for i in range(2):
        classifier = "W8" + str(i)
        sev_model.add_agent(classifier=classifier,
                            occ=7,
                            mr=30,
                            stat=0)

    
    ## 9th floor
    # transfers
    for i in range(2):
        classifier = "T9" + str(i)
        sev_model.add_agent(classifier=classifier,
                            occ=1,
                            mr=40,
                            stat=0)
        
    # operational therapists
    for i in range(3):
        classifier = "O9" + str(i)
        sev_model.add_agent(classifier=classifier,
                            occ=5,
                            mr=43,
                            stat = 0)
        
    # physical therapists
    for i in range(26):
        classifier = "H9" + str(i)
        sev_model.add_agent(classifier=classifier,
                            occ=6,
                            mr=42,
                            stat = 0)
        
    # cleaner
    classifier = "W90"
    sev_model.add_agent(classifier=classifier,
                        occ=7,
                        mr=40,
                        stat = 0)
    
    
    ## 10th floor
    # nurses
    for i in range(12):
        classifier = "N1" + str(i)
        sev_model.add_agent(classifier=classifier,
                            occ=0,
                            mr=51,
                            stat=0)

    # transfers
    for i in range(2):
        classifier = "T1" + str(i)
        sev_model.add_agent(classifier=classifier,
                            occ=1,
                            mr=50,
                            stat=0)

    # patients
    for i in range(2):
        for j in range(8):
            classifier = "P1" + str(8*i + j)
            sev_model.add_agent(classifier=classifier,
                                occ=2,
                                mr=i+52,
                                stat=0)
    for j in range(2):
        classifier = "P1" + str(16 + j)
        sev_model.add_agent(classifier=classifier,
                            occ=2,
                            mr=54,
                            stat=0)

    # caregivers
    for i in range(2):
        for j in range(8):
            classifier = "C1" + str(8*i + j)
            sev_model.add_agent(classifier=classifier,
                                occ=3,
                                mr=i+52,
                                stat=0)
    for j in range(2):
        classifier = "C1" + str(16 + j)
        sev_model.add_agent(classifier=classifier,
                            occ=3,
                            mr=54,
                            stat=0)
            
    # cleaners
    for i in range(2):
        classifier = "W1" + str(i)
        sev_model.add_agent(classifier=classifier,
                            occ=7,
                            mr=50,
                            stat=0)
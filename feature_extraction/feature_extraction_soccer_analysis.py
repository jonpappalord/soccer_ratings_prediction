import pandas as pd
from pprint import pprint

from tqdm import tqdm

'''
Creation of the big match ids set
'''
bigMatch = set()
#Juventus
bigMatch.add(3159)
#Inter
bigMatch.add(3161)
#Roma
bigMatch.add(3158)
#Milan
bigMatch.add(3157)
#Lazio
bigMatch.add(3162)
#Napoli
bigMatch.add(3187)
#Atalanta
bigMatch.add(3172)


def playerMatchExtraction(dfmatc):
    """
    Extract from match JSON file (wyscout dataset) all the informations regarding gameweek, player, player entered, red card
    yellow card, minutes played ecc..

    Parameters
    ----------
    dfmatc : Matches DataFrame
        The dataset from json file of Wyscout matches.

    Returns
    -------
    pandas DataFrame
        A dataframe with all the informations mentioned above.
    """
    listGameWeek = []
    listWyIdMatch = []
    listTeamId = []
    listPlayerId = []
    listMinutesPlayed = []
    listYellowCard = []
    listRedCard = []
    listGoals = []

    # for each match
    for el in dfmatc.values:
        # for each team
        for el1 in el[10]:
            # in titol we have the real formation that came insde the field
            titol = el[10][el1]['formation']['lineup']
            # sobstitutions are the player that entered
            substitutions = el[10][el1]['formation']['substitutions']
            # bench are all the player in the bench
            bench = el[10][el1]['formation']['bench']

            # for each player titolare we need to check if it came against a sostitution
            for playerTitolare in titol:
                minutes = 0
                # we look in the substititions
                for playerEntered in substitutions:
                    # if ids match
                    if (playerEntered['playerOut'] == playerTitolare['playerId']):
                        # we assign a value to minutes
                        minutes = playerEntered['minute']
                # match found if we assigned a value to minutes
                if (minutes != 0):
                    listMinutesPlayed.append(minutes)
                else:
                    listMinutesPlayed.append(90)
                listGameWeek.append(el[4])
                listTeamId.append(int(el1.replace('\'', '')))
                listWyIdMatch.append(el[13])
                listPlayerId.append(playerTitolare['playerId'])
                listRedCard.append(playerTitolare['redCards'])
                listYellowCard.append(playerTitolare['yellowCards'])
                listGoals.append(playerTitolare['goals'])
            # for each entered player
            for playerEntered in substitutions:
                # we look inside the bench
                for playerBench in bench:
                    # the one entered has the same id
                    if (playerBench['playerId'] == playerEntered['playerIn']):
                        # substituitions
                        listGameWeek.append(el[4])
                        listTeamId.append(int(el1.replace('\'', '')))
                        listWyIdMatch.append(el[13])
                        listPlayerId.append(playerEntered['playerIn'])
                        listMinutesPlayed.append(90 - playerEntered['minute'])
                        listRedCard.append(playerBench['redCards'])
                        listYellowCard.append(playerBench['yellowCards'])
                        listGoals.append(playerBench['goals'])
    newYellowCardList = []
    newRedCardList = []
    for yellow in listYellowCard:
        if (int(yellow) > 0):
            newYellowCardList.append(1)
        else:
            newYellowCardList.append(0)
    listYellowCard = newYellowCardList

    for red in listRedCard:
        if (int(red) > 0):
            newRedCardList.append(1)
        else:
            newRedCardList.append(0)
    listRedCard = newRedCardList

    return pd.DataFrame(
        {'match_id': listWyIdMatch, 'gameweek': listGameWeek, 'team_id': listTeamId, 'playerId': listPlayerId,
         'minutes_played': listMinutesPlayed, 'red_card': listRedCard, 'yellow_card': listYellowCard,
         'goals': listGoals})


def checkSpecificTagInsideList(tags, code):
    """
    This function, using the tag list of event, return a boolean that says if a tag is inside or not inside the tags list

    Parameters
    ----------
    tags : List of dictionary that contain id related to the event

    code : The code is needed to be checked inside the tag list

    Returns
    -------
    boolean : True if the element is inside the list
              False if the lement is NOT inside the list
    """
    # for each dict in the list
    for checkTag in tags:
        # if the code is inside the dictionary values
        if (code in checkTag.values()):
            # return false
            return True
    return False

def percentage(percent, whole):
    return int((percent * whole) / 100.0)

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def attackToRight(listOfPosition):
    '''
    Given the first event of a match, start from passes position define if the team attack on left or right

    ##RECALL FROM EACH QUALITY FEATURE COMPUTATION
    Params:
        List of position at 0 the starting point of the pass
                            1 the ending point of the pass
    Return
        True:  the movement of the ball go to the left, so the team attack on right
        False: the movement of the ball go to the right, so the team attack on left
    '''
    x1 = listOfPosition[0]['x']
    x2 = listOfPosition[1]['x']
    if (x2 > x1):
        return False
    else:
        return True


def helperForUpgradingElement(teamId, teamIn1H, half, position, right, kind):
    '''
    This function helps in identify if the event refers to the team that attack on right or on left in order
    to retrive the value of danger.

    ##RECALL FROM EACH QUALITY FEATURE COMPUTATION

    Params:
        teamId = the event team id
        teamIn1H = the first team that touche the first event of the match
        half = the half of the event
        position = the list of coordinates of the event
        right = true if the team attack on right, false on left
        kind = true if the event is a positive event check on the enemy side, false if is negative so need to check in own side

    Return
        the dangerous value of the event
    '''
    ##N.B. THE TEAMS ATTACK ALWAYS ON THE RIGHT, THE KEY IS THE KIND OF ACTION THAT DEFINES IN WHICH PART OF THE PITCH WE LOOKED FOR
    right = True
    # dangerous computation
    # first possession
    if (teamId == teamIn1H):
        # first half
        if (half == '1H'):
            return checkPositioningInThePitch(position, right, kind)
        # second half, change attack size
        else:
            return checkPositioningInThePitch(position, right, kind)
    # non possession
    else:
        # first half
        if (half == '1H'):
            return checkPositioningInThePitch(position, right, kind)
        # second half
        else:
            return checkPositioningInThePitch(position, right, kind)


def checkPositionToRightPitch(i, l):
    if (i == 95 and l > 29 and l < 35):
        return 1
    elif (i == 95 and l <= 29 and l >= 26):
        return 0.8
    elif (i == 95 and l <= 38 and l >= 35):
        return 0.8
    elif (i == 95 and l <= 26):
        return 0.1
    elif (i == 95 and l > 38):
        return 0.1

    elif (i == 97 and l <= 33 and l >= 29):
        return 0.7
    elif (i == 97 and l <= 36 and l > 33):
        return 0.5
    elif (i == 97 and l < 29 and l >= 26):
        return 0.5
    elif (i == 97 and l <= 39 and l > 36):
        return 0.3
    elif (i == 97 and l < 26 and l >= 24):
        return 0.3
    elif (i == 97 and l >= 40):
        return 0.1
    elif (i == 97 and l < 24):
        return 0.1

    elif (i == 93 and l >= 30 and l < 36):
        return 1
    elif (i == 93 and l <= 29 and l > 27):
        return 0.8
    elif (i == 93 and l >= 36 and l < 39):
        return 0.8
    elif (i == 93 and l <= 27):
        return 0.1
    elif (i == 93 and l >= 39):
        return 0.1

    elif (i == 91 and l >= 28 and l <= 36):
        return 1
    elif (i == 91 and l >= 27 and l <= 28):
        return 0.8
    elif (i == 91 and l >= 36 and l <= 37):
        return 0.8
    elif (i == 91 and l >= 25 and l <= 29):
        return 0.7
    elif (i == 91 and l >= 37 and l <= 38):
        return 0.7
    elif (i == 91 and l >= 25 and l <= 29):
        return 0.7
    elif (i == 91 and l >= 37 and l <= 38):
        return 0.7
    elif (i == 91 and l >= 23 and l <= 25):
        return 0.5
    elif (i == 91 and l >= 38 and l <= 40):
        return 0.5
    elif (i == 91 and l >= 40 and l <= 42):
        return 0.1
    elif (i == 91 and l < 23 and l >= 22):
        return 0.1
    elif (i == 91 and l > 42 and l <= 43):
        return 0.5
    elif (i == 91 and l < 22 and l >= 21):
        return 0.5
    elif (i == 91 and l > 40 and l <= 41):
        return 0.3
    elif (i == 91 and l < 21 and l >= 20):
        return 0.3
    elif (i == 91 and l < 20):
        return 0.1
    elif (i == 91 and l > 41):
        return 0.1

    elif (i == 89 and l >= 24 and l <= 40):
        return 0.5
    elif (i == 89 and l >= 22 and l < 24):
        return 0.3
    elif (i == 89 and l >= 40 and l <= 42):
        return 0.3
    elif (i == 89 and l < 22):
        return 0.1
    elif (i == 89 and l > 42):
        return 0.1

    elif (i == 87 and l >= 24 and l <= 40):
        return 0.5
    elif (i == 87 and l >= 22 and l < 24):
        return 0.3
    elif (i == 87 and l >= 40 and l <= 42):
        return 0.3
    elif (i == 87 and l < 22):
        return 0.1
    elif (i == 87 and l > 42):
        return 0.1

    elif (i == 85 and l >= 26 and l <= 38):
        if (l == 34):
            return 0.8
        elif (l == 32):
            return 0.1
        else:
            return 0.5
    elif (i == 85 and l >= 24 and l < 26):
        return 0.3
    elif (i == 85 and l >= 38 and l <= 40):
        return 0.3
    elif (i == 85 and l < 24):
        return 0.1
    elif (i == 85 and l > 40):
        return 0.1

    elif (i == 83 and l >= 26 and l <= 38):
        return 0.3
    elif (i == 83 and l < 26):
        return 0.1
    elif (i == 83 and l > 38):
        return 0.1

    elif (i == 81 and l >= 29 and l <= 33):
        return 0.3
    elif (i == 81 and l < 29):
        return 0.1
    elif (i == 81 and l > 33):
        return 0.1

    elif (i < 60 or i > 97):
        return 0
    elif (l < 12 or l > 51):
        return 0
    else:
        return 0.1


def checkPositionToLeftPitch(i, l):
    if (i == 5 and l > 29 and l < 35):
        return 1
    elif (i == 5 and l <= 29 and l >= 26):
        return 0.8
    elif (i == 5 and l <= 38 and l >= 35):
        return 0.8
    elif (i == 5 and l <= 26):
        return 0.1
    elif (i == 5 and l > 38):
        return 0.1

    elif (i == 3 and l <= 33 and l >= 29):
        return 0.7
    elif (i == 3 and l <= 36 and l > 33):
        return 0.5
    elif (i == 3 and l < 29 and l >= 26):
        return 0.5
    elif (i == 3 and l <= 39 and l > 36):
        return 0.3
    elif (i == 3 and l < 26 and l >= 24):
        return 0.3
    elif (i == 3 and l >= 40):
        return 0.1
    elif (i == 3 and l < 24):
        return 0.1

    elif (i == 7 and l >= 30 and l < 36):
        return 1
    elif (i == 7 and l <= 29 and l > 27):
        return 0.8
    elif (i == 7 and l >= 36 and l < 39):
        return 0.8
    elif (i == 7 and l <= 27):
        return 0.1
    elif (i == 7 and l >= 39):
        return 0.1

    elif (i == 9 and l >= 28 and l <= 36):
        return 1
    elif (i == 9 and l >= 27 and l <= 28):
        return 0.8
    elif (i == 9 and l >= 36 and l <= 37):
        return 0.8
    elif (i == 9 and l >= 25 and l <= 29):
        return 0.7
    elif (i == 9 and l >= 37 and l <= 38):
        return 0.7
    elif (i == 9 and l >= 25 and l <= 29):
        return 0.7
    elif (i == 9 and l >= 37 and l <= 38):
        return 0.7
    elif (i == 9 and l >= 23 and l <= 25):
        return 0.5
    elif (i == 9 and l >= 38 and l <= 40):
        return 0.5
    elif (i == 9 and l >= 40 and l <= 42):
        return 0.1
    elif (i == 9 and l < 23 and l >= 22):
        return 0.1
    elif (i == 9 and l > 42 and l <= 43):
        return 0.5
    elif (i == 9 and l < 22 and l >= 21):
        return 0.5
    elif (i == 9 and l > 40 and l <= 41):
        return 0.3
    elif (i == 9 and l < 21 and l >= 20):
        return 0.3
    elif (i == 9 and l < 20):
        return 0.1
    elif (i == 9 and l > 41):
        return 0.1

    elif (i == 11 and l >= 24 and l <= 40):
        return 0.5
    elif (i == 11 and l >= 22 and l < 24):
        return 0.3
    elif (i == 11 and l >= 40 and l <= 42):
        return 0.3
    elif (i == 11 and l < 22):
        return 0.1
    elif (i == 11 and l > 42):
        return 0.1

    elif (i == 13 and l >= 24 and l <= 40):
        return 0.5
    elif (i == 13 and l >= 22 and l < 24):
        return 0.3
    elif (i == 13 and l >= 40 and l <= 42):
        return 0.3
    elif (i == 13 and l < 22):
        return 0.1
    elif (i == 13 and l > 42):
        return 0.1

    elif (i == 15 and l >= 26 and l <= 38):
        if (l == 34):
            return 0.8
        elif (l == 32):
            return 0.1
        else:
            return 0.5
    elif (i == 15 and l >= 24 and l < 26):
        return 0.3
    elif (i == 15 and l >= 38 and l <= 40):
        return 0.3
    elif (i == 15 and l < 24):
        return 0.1
    elif (i == 15 and l > 40):
        return 0.1

    elif (i == 17 and l >= 26 and l <= 38):
        return 0.3
    elif (i == 17 and l < 26):
        return 0.1
    elif (i == 17 and l > 38):
        return 0.1

    elif (i == 19 and l >= 29 and l <= 33):
        return 0.3
    elif (i == 19 and l < 29):
        return 0.1
    elif (i == 19 and l > 33):
        return 0.1

    elif (i > 40 or i < 3):
        return 0
    elif (l < 12 or l > 51):
        return 0
    else:
        return 0.1


def checkPositioningInThePitch(listOfPostionings, right, positive):
    '''
    Take the list of dictionary for postioning inside the field.

    Postion are from 0 to 100 position.

    right means if the team attack on left or attack on right. Set right to true means that the team attack from left to right

    postive means if the action to check means a positive action or a negative action (pass complete, pass failed)

    '''
    i = 0
    l = 0
    # if the lenght del postioning maggiore a uno faccio il calcolo
    # the index says the starting or arriving point
    if (len(listOfPostionings) > 1):
        i = listOfPostionings[0]['x']
        l = listOfPostionings[0]['y']
    else:
        i = listOfPostionings[0]['x']
        l = listOfPostionings[0]['y']
    # we rescale the y axis
    l = percentage(l, 64)

    # we rescale the x axis to values that are close to
    if ((i % 2) == 0):
        i = i + 1

    # the event is from a team that attack to right
    if (right):
        # the event is a positive event, so we look into the right of the pitch
        if (positive):
            return checkPositionToRightPitch(i, l)
        # the event is a negative event, so we look into the left of the pitch
        else:
            return checkPositionToLeftPitch(i, l)
    # attack on left
    else:
        # the event is a positive event, so we look into the left of the pitch
        if (positive):
            return checkPositionToLeftPitch(i, l)
        # the event is a negative event, so we look into the right of the pitch
        else:
            return checkPositionToRightPitch(i, l)


def computeTotalPasses(df, dfEvent):
    """
    For each player in df look the number of air duel inside a match

    Parameters
    ----------
    df : match_id	gameweek	team_id	playerId	minutes_played	red_card	yellow_card	goals
        The dataset extracted with playerMatchExtraction from json file of Wyscout matches.

    dfEvent : The dataset that contain event of matches

    Returns
    -------
    df Dataframe
        The input data frame with the passes features included
    """
    listTotalPasses = []
    listTotalCompletedPasses = []
    listTotalFailedPasses = []
    listTotalKeyPasses = []
    listTotalAssistPasses = []

    # dangerous features
    listDangerousCompletePasses = []
    listDangerousFailedPasses = []
    listDangerousKeyPasses = []
    listDangerousAssistPasses = []
    listDangerousPlayerId = []
    listDangerousMatch = []

    # Initial call to print 0% progress
    printProgressBar(0, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)

    # for each player in the dataset
    for index, playerMatch in df.iterrows():
        # Update Progress Bar
        printProgressBar(index, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)
        # the number of air duel instantiated to 0 for each player
        counterPlayer = 0
        counterCompletePasses = 0
        counterFailedPasses = 0
        counterTotalKeyPasses = 0
        counterTotalAssistPasses = 0

        # dangerous features
        dangerousCompletePasses = 0
        dangerousFailedPasses = 0
        dangerousKeyPasses = 0
        dangerousAssistPasses = 0

        # we extract these information in order to reduce the size of the dataset to scroll
        matchId = playerMatch[0]
        playerId = playerMatch[3]
        datasetToScroll = dfEvent.loc[(dfEvent.matchId == matchId) & (dfEvent.playerId == playerId)]

        # dangerous stuff
        teamIn1H = dfEvent[dfEvent['matchId'] == matchId]['teamId'].iloc[0]
        right = attackToRight(dfEvent[dfEvent['matchId'] == matchId]['positions'].iloc[0])

        # for each event in the subset
        for passe in datasetToScroll.values:

            # we count the number of passes
            if (passe[1] == 'Pass'):
                # passage so we add it as a counter to total number
                counterPlayer = counterPlayer + 1

                # we check for id 1801 (accurate id), accurate passes means complete passes
                if (checkSpecificTagInsideList(passe[10], 1801)):
                    counterCompletePasses = counterCompletePasses + 1
                    dangerousCompletePasses = dangerousCompletePasses + helperForUpgradingElement(passe[11], teamIn1H,
                                                                                                  passe[5], passe[7],
                                                                                                  right, True)

                # we check for id 1802 (not accurate id), not accurate passes means failed passes
                if (checkSpecificTagInsideList(passe[10], 1802)):
                    counterFailedPasses = counterFailedPasses + 1
                    dangerousFailedPasses = dangerousFailedPasses + helperForUpgradingElement(passe[11], teamIn1H,
                                                                                              passe[5], passe[7], right,
                                                                                              False)

                # we check for id 302 (key pass id)
                if (checkSpecificTagInsideList(passe[10], 302)):
                    counterTotalKeyPasses = counterTotalKeyPasses + 1
                    dangerousKeyPasses = dangerousKeyPasses + helperForUpgradingElement(passe[11], teamIn1H, passe[5],
                                                                                        passe[7], right, True)

                # we check for id 301 (assist pass id)
                if (checkSpecificTagInsideList(passe[10], 301)):
                    counterTotalAssistPasses = counterTotalAssistPasses + 1
                    dangerousAssistPasses = dangerousAssistPasses + helperForUpgradingElement(passe[11], teamIn1H,
                                                                                              passe[5], passe[7], right,
                                                                                              True)

        listTotalPasses.append(counterPlayer)
        listTotalCompletedPasses.append(counterCompletePasses)
        listTotalFailedPasses.append(counterFailedPasses)
        listTotalKeyPasses.append(counterTotalKeyPasses)
        listTotalAssistPasses.append(counterTotalAssistPasses)

        # dangerous part
        listDangerousAssistPasses.append(dangerousAssistPasses)
        listDangerousCompletePasses.append(dangerousCompletePasses)
        listDangerousFailedPasses.append(dangerousFailedPasses)
        listDangerousKeyPasses.append(dangerousKeyPasses)
        listDangerousPlayerId.append(playerId)
        listDangerousMatch.append(matchId)

    df['total_passes'] = listTotalPasses
    df['completed_passes'] = listTotalCompletedPasses
    df['failed_passes'] = listTotalFailedPasses
    df['key_passes'] = listTotalKeyPasses
    df['assist_passes'] = listTotalAssistPasses

    dangerousDF = pd.DataFrame({'id': listDangerousPlayerId,
                                'match': listDangerousMatch,
                                'dangerous_key_passes': listDangerousKeyPasses,
                                'dangerous_assist_passes': listDangerousAssistPasses,
                                'dangerous_failed_passes': listDangerousFailedPasses,
                                'dangerous_complete_passes': listDangerousCompletePasses})
    return df, dangerousDF


def computeTotalNumberCross(df, dfEvent):
    """
    For each player in df look the number of air duel inside a match

    Parameters
    ----------
    df : match_id	gameweek	team_id	playerId	minutes_played	red_card	yellow_card	goals
        The dataset extracted with playerMatchExtraction from json file of Wyscout matches.

    dfEvent : The dataset that contain event of matches

    Returns
    -------
    df Dataframe
        The input data frame with the cross features included
    """
    listTotalCross = []
    listTotalNumberOfCompletedCross = []
    listTotalNumberOfFailedCross = []
    listTotalNumberOfAssistCross = []

    # dangerous features
    listDangerousCompletedCross = []
    listDangerousFailedCross = []
    listDangerousAssistCross = []
    listDangerId = []
    listDangerMatch = []

    # Initial call to print 0% progress
    printProgressBar(0, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)

    # for each player in the dataset
    for index, playerMatch in df.iterrows():
        printProgressBar(index, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)
        # the number of air duel instantiated to 0 for each player
        counterPlayer = 0
        counterCompletedCross = 0
        counterFailedCross = 0
        counterAssistCross = 0

        # dangerous features
        dangerousCompletedCross = 0
        dangerousFailedCross = 0
        dangerousAssistCross = 0

        # we extract these information in order to reduce the size of the dataset to scroll
        matchId = playerMatch[0]
        playerId = playerMatch[3]
        datasetToScroll = dfEvent.loc[(dfEvent.matchId == matchId) & (dfEvent.playerId == playerId)]

        # dangerous stuff
        teamIn1H = dfEvent[dfEvent['matchId'] == matchId]['teamId'].iloc[0]
        right = attackToRight(dfEvent[dfEvent['matchId'] == matchId]['positions'].iloc[0])

        # for each event in the subset
        for cross in datasetToScroll.values:

            # we look for pass, more in deep crosses
            if (cross[1] == 'Pass' and cross[9] == 'Cross'):
                counterPlayer = counterPlayer + 1
                # we check cross that are accurate (id 1801)
                if (checkSpecificTagInsideList(cross[10], 1801)):
                    counterCompletedCross = counterCompletedCross + 1
                    dangerousCompletedCross = dangerousCompletedCross + helperForUpgradingElement(cross[11], matchId,
                                                                                                  cross[5], cross[7],
                                                                                                  right, True)
                # we check cross that are inaccurate (id 1802)
                if (checkSpecificTagInsideList(cross[10], 1802)):
                    counterFailedCross = counterFailedCross + 1
                    dangerousFailedCross = dangerousFailedCross + helperForUpgradingElement(cross[11], matchId,
                                                                                            cross[5], cross[7], right,
                                                                                            False)
                # we check cross that are assist (id 301)
                if (checkSpecificTagInsideList(cross[10], 301)):
                    counterAssistCross = counterAssistCross + 1
                    dangerousAssistCross = dangerousAssistCross + helperForUpgradingElement(cross[11], matchId,
                                                                                            cross[5], cross[7], right,
                                                                                            True)

            # we look also for free kick crosses
            if (cross[1] == 'Free Kick' and cross[9] == 'Free kick cross'):
                counterPlayer = counterPlayer + 1
                # we check cross that are accurate (id 1801)
                if (checkSpecificTagInsideList(cross[10], 1801)):
                    counterCompletedCross = counterCompletedCross + 1
                    dangerousCompletedCross = dangerousCompletedCross + helperForUpgradingElement(cross[11], matchId,
                                                                                                  cross[5], cross[7],
                                                                                                  right, True)
                # we check cross that are inaccurate (id 1802)
                if (checkSpecificTagInsideList(cross[10], 1802)):
                    counterFailedCross = counterFailedCross + 1
                    dangerousFailedCross = dangerousFailedCross + helperForUpgradingElement(cross[11], matchId,
                                                                                            cross[5], cross[7], right,
                                                                                            False)
                # we check cross that are assist (id 301)
                if (checkSpecificTagInsideList(cross[10], 301)):
                    counterAssistCross = counterAssistCross + 1
                    dangerousAssistCross = dangerousAssistCross + helperForUpgradingElement(cross[11], matchId,
                                                                                            cross[5], cross[7], right,
                                                                                            True)

        listTotalCross.append(counterPlayer)
        listTotalNumberOfCompletedCross.append(counterCompletedCross)
        listTotalNumberOfFailedCross.append(counterFailedCross)
        listTotalNumberOfAssistCross.append(counterAssistCross)

        listDangerousAssistCross.append(dangerousAssistCross)
        listDangerousCompletedCross.append(dangerousCompletedCross)
        listDangerousFailedCross.append(dangerousFailedCross)
        listDangerId.append(playerId)
        listDangerMatch.append(matchId)

    df['total_cross'] = listTotalCross
    df['completed_cross'] = listTotalNumberOfCompletedCross
    df['failed_cross'] = listTotalNumberOfFailedCross
    df['assist_cross'] = listTotalNumberOfAssistCross

    dangerousDF = pd.DataFrame({'id': listDangerId,
                                'match': listDangerMatch,
                                'dangerous_assist_cross': listDangerousAssistCross,
                                'dangerous_failed_cross': listDangerousFailedCross,
                                'dangerous_complete_cross': listDangerousCompletedCross})
    return df, dangerousDF


def computeTotalNumberTackels(df, dfEvent):
    """
    For each player in df look the number of air duel inside a match

    Parameters
    ----------
    df : match_id	gameweek	team_id	playerId	minutes_played	red_card	yellow_card	goals
        The dataset extracted with playerMatchExtraction from json file of Wyscout matches.

    dfEvent : The dataset that contain event of matches

    Returns
    -------
    df Dataframe
        The input data frame with the tackels features included
    """
    listTotalTackels = []
    listCompletedTackels = []
    listFailedTackels = []

    listDangerousCompletedTackels = []
    listDangerousFailedTackels = []
    listDangerousId = []
    listMatchId = []

    # Initial call to print 0% progress
    printProgressBar(0, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)

    # for each player in the dataset
    for index, playerMatch in df.iterrows():

        printProgressBar(index, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)
        # the number of air duel instantiated to 0 for each player
        counterPlayer = 0
        counterCompleteTackels = 0
        counterFailedTackels = 0

        dangerousCompletedTackels = 0
        dangerousFailedTackels = 0

        # we extract these information in order to reduce the size of the dataset to scroll
        matchId = playerMatch[0]
        playerId = playerMatch[3]
        datasetToScroll = dfEvent.loc[(dfEvent.matchId == matchId) & (dfEvent.playerId == playerId)]

        # dangerous stuff
        teamIn1H = dfEvent[dfEvent['matchId'] == matchId]['teamId'].iloc[0]
        right = attackToRight(dfEvent[dfEvent['matchId'] == matchId]['positions'].iloc[0])

        # for each event in the subset
        for tack in datasetToScroll.values:
            # we look for ground duel
            if (tack[1] == 'Duel' and tack[9] != 'Air duel'):

                # we check that code 1601 (sliding tackels is inside the tags list)
                if (checkSpecificTagInsideList(tack[10], 1601)):
                    counterPlayer = counterPlayer + 1
                    # we look for completed (N.B. WON OR ACCURATE????) ones (id 703)
                    if (checkSpecificTagInsideList(tack[10], 703) or checkSpecificTagInsideList(tack[10], 702)):
                        counterCompleteTackels = counterCompleteTackels + 1
                        dangerousCompletedTackels = dangerousCompletedTackels + helperForUpgradingElement(tack[11],
                                                                                                          matchId,
                                                                                                          tack[5],
                                                                                                          tack[7],
                                                                                                          right, False)
                        # we look for failed (N.B. LOST OR INNACURATE???) ones (id 701)
                    if (checkSpecificTagInsideList(tack[10], 701)):
                        counterFailedTackels = counterFailedTackels + 1
                        dangerousFailedTackels = dangerousFailedTackels + helperForUpgradingElement(tack[11], matchId,
                                                                                                    tack[5], tack[7],
                                                                                                    right, False)

        listTotalTackels.append(counterPlayer)
        listCompletedTackels.append(counterCompleteTackels)
        listFailedTackels.append(counterFailedTackels)
        listDangerousCompletedTackels.append(dangerousCompletedTackels)
        listDangerousFailedTackels.append(dangerousFailedTackels)
        listDangerousId.append(playerId)
        listMatchId.append(matchId)

    df['total_tackels'] = listTotalTackels
    df['completed_tackels'] = listCompletedTackels
    df['failed_tackels'] = listFailedTackels

    dangerousDF = pd.DataFrame({'id': listDangerousId,
                                'match': listMatchId,
                                'dangerous_failed_tackels': listDangerousFailedTackels,
                                'dangerous_complete_tackels': listDangerousCompletedTackels})

    return df, dangerousDF


def computeTotalNumberDribbling(df, dfEvent):
    """
    For each player in df look the number of DRIBBLING inside a match

    Parameters
    ----------
    df : match_id	gameweek	team_id	playerId	minutes_played	red_card	yellow_card	goals
        The dataset extracted with playerMatchExtraction from json file of Wyscout matches.

    dfEvent : The dataset that contain event of matches

    Returns
    -------
    df Dataframe
        The input data frame with the dribling features included
    """
    listTotalDribblings = []
    listCompletedDribbling = []
    listFailedDribbling = []

    listDangerousCompletedDribbling = []
    listDangerousFailedDribbling = []
    listDangrousId = []
    listDangerousMatch = []

    # Initial call to print 0% progress
    printProgressBar(0, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)

    # for each player in the dataset
    for index, playerMatch in df.iterrows():
        printProgressBar(index, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)
        # the number of air duel instantiated to 0 for each player
        counterPlayer = 0
        counterCompleteDribbling = 0
        counterFailedDribbling = 0

        dangerousCompletedDribbling = 0
        dangerousFailedDribbling = 0

        # we extract these information in order to reduce the size of the dataset to scroll
        matchId = playerMatch[0]
        playerId = playerMatch[3]
        datasetToScroll = dfEvent.loc[(dfEvent.matchId == matchId) & (dfEvent.playerId == playerId)]

        # dangerous stuff
        teamIn1H = dfEvent[dfEvent['matchId'] == matchId]['teamId'].iloc[0]
        right = attackToRight(dfEvent[dfEvent['matchId'] == matchId]['positions'].iloc[0])

        # for each event in the subset
        for drib in datasetToScroll.values:
            # we look for attacking ground duel
            if (drib[9] == 'Ground attacking duel'):

                # we check that code 503 or 504 is inside the tag (take on l and take on r)
                # N.B. DOUBT ABOUT HAVING ALSO 502 AND 501
                if (checkSpecificTagInsideList(drib[10], 503) or checkSpecificTagInsideList(drib[10], 504)):
                    counterPlayer = counterPlayer + 1

                    # we look for completed (N.B. WON OR ACCURATE????) ones (id 703)
                    if (checkSpecificTagInsideList(drib[10], 703) or checkSpecificTagInsideList(drib[10], 702)):
                        counterCompleteDribbling = counterCompleteDribbling + 1
                        dangerousCompletedDribbling = dangerousCompletedDribbling + helperForUpgradingElement(drib[11],
                                                                                                              matchId,
                                                                                                              drib[5],
                                                                                                              drib[7],
                                                                                                              right,
                                                                                                              True)

                    # we look for failed (N.B. LOST OR INNACURATE???) ones (id 701)
                    if (checkSpecificTagInsideList(drib[10], 701)):
                        counterFailedDribbling = counterFailedDribbling + 1
                        dangerousFailedDribbling = dangerousFailedDribbling + helperForUpgradingElement(drib[11],
                                                                                                        matchId,
                                                                                                        drib[5],
                                                                                                        drib[7], right,
                                                                                                        False)

        listTotalDribblings.append(counterPlayer)
        listCompletedDribbling.append(counterCompleteDribbling)
        listFailedDribbling.append(counterFailedDribbling)
        listDangerousCompletedDribbling.append(dangerousCompletedDribbling)
        listDangerousFailedDribbling.append(dangerousFailedDribbling)
        listDangrousId.append(playerId)
        listDangerousMatch.append(matchId)
    df['total_dribblings'] = listTotalDribblings
    df['completed_dribblings'] = listCompletedDribbling
    df['failed_dribblings'] = listFailedDribbling

    dangerousDF = pd.DataFrame({'id': listDangrousId,
                                'match': listDangerousMatch,
                                'dangerous_failed_dribblings': listDangerousFailedDribbling,
                                'dangerous_complete_dribblings': listDangerousCompletedDribbling})

    return df, dangerousDF


def computeTotalAirDuel(df, dfEvent):
    """
    For each player in df look the number of air duel inside a match

    Parameters
    ----------
    df : match_id	gameweek	team_id	playerId	minutes_played	red_card	yellow_card	goals
        The dataset extracted with playerMatchExtraction from json file of Wyscout matches.

    dfEvent : The dataset that contain event of matches

    Returns
    -------
    df Dataframe
        The input data frame with the headed duel features included
    """
    listAirDuel = []
    listCompleteAirDuel = []
    listFailedAirDuel = []

    listDangerousCompleteAirDuel = []
    listDangerousFailedAirDuel = []
    listDangerousId = []
    listDangerousMatch = []

    # Initial call to print 0% progress
    printProgressBar(0, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)

    # for each player in the dataset
    for index, playerMatch in df.iterrows():
        printProgressBar(index, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)
        # the number of air duel instantiated to 0 for each player
        counterPlayer = 0
        counterCompleteAirDuel = 0
        counterFailedAirDuel = 0

        dangerousFailedAirDuel = 0
        dangerousCompleteAirDuel = 0

        # we extract these information in order to reduce the size of the dataset to scroll
        matchId = playerMatch[0]
        playerId = playerMatch[3]
        datasetToScroll = dfEvent.loc[(dfEvent.matchId == matchId) & (dfEvent.playerId == playerId)]

        # dangerous stuff
        teamIn1H = dfEvent[dfEvent['matchId'] == matchId]['teamId'].iloc[0]
        right = attackToRight(dfEvent[dfEvent['matchId'] == matchId]['positions'].iloc[0])

        # for each event in the subset
        for head in datasetToScroll.values:

            # we count the number of air duel
            if (head[9] == 'Air duel'):
                counterPlayer = counterPlayer + 1

                # we look for completed (N.B. WON OR ACCURATE????) ones (id 703)
                if (checkSpecificTagInsideList(head[10], 703) or checkSpecificTagInsideList(head[10], 702)):
                    counterCompleteAirDuel = counterCompleteAirDuel + 1
                    dangerousCompleteAirDuel = dangerousCompleteAirDuel + helperForUpgradingElement(head[11], matchId,
                                                                                                    head[5], head[7],
                                                                                                    right, False)
                    dangerousCompleteAirDuel = dangerousCompleteAirDuel + helperForUpgradingElement(head[11], matchId,
                                                                                                    head[5], head[7],
                                                                                                    right, True)

                # we look for failed (N.B. LOST OR INNACURATE???) ones (id 701)
                if (checkSpecificTagInsideList(head[10], 701)):
                    counterFailedAirDuel = counterFailedAirDuel + 1
                    dangerousFailedAirDuel = dangerousFailedAirDuel + helperForUpgradingElement(head[11], matchId,
                                                                                                head[5], head[7], right,
                                                                                                False)
                    dangerousFailedAirDuel = dangerousFailedAirDuel + helperForUpgradingElement(head[11], matchId,
                                                                                                head[5], head[7], right,
                                                                                                True)
        listAirDuel.append(counterPlayer)
        listCompleteAirDuel.append(counterCompleteAirDuel)
        listFailedAirDuel.append(counterFailedAirDuel)
        listDangerousCompleteAirDuel.append(dangerousCompleteAirDuel)
        listDangerousFailedAirDuel.append(dangerousFailedAirDuel)
        listDangerousId.append(playerId)
        listDangerousMatch.append(matchId)
    df['headed_duel'] = listAirDuel
    df['complete_headed_duel'] = listCompleteAirDuel
    df['failed_headed_duel'] = listFailedAirDuel

    dangerousDF = pd.DataFrame({'id': listDangerousId,
                                'match': listDangerousMatch,
                                'dangerous_failed_air_duel': listDangerousFailedAirDuel,
                                'dangerous_complete_dribblings': listDangerousCompleteAirDuel})

    return df, dangerousDF


def computeTotalNumberInterceptions(df, dfEvent):
    """
    For each player in df look the number of Interceptions inside a match

    Parameters
    ----------
    df : match_id	gameweek	team_id	playerId	minutes_played	red_card	yellow_card	goals
        The dataset extracted with playerMatchExtraction from json file of Wyscout matches.

    dfEvent : The dataset that contain event of matches

    Returns
    -------
    df Dataframe
        The input data frame with the Interceptions features included
    """
    listTotalInterceptions = []
    listCompletedInterceptions = []

    listDangerousCompletedInterceptions = []
    listDangerousId = []
    listDangerousMatch = []

    # Initial call to print 0% progress
    printProgressBar(0, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)

    # for each player in the dataset
    for index, playerMatch in df.iterrows():
        printProgressBar(index, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)
        # the number of air duel instantiated to 0 for each player
        counterPlayer = 0
        counterCompletedInterceptions = 0
        counterDangerousCompletedInterceptions = 0

        # we extract these information in order to reduce the size of the dataset to scroll
        matchId = playerMatch[0]
        playerId = playerMatch[3]
        datasetToScroll = dfEvent.loc[(dfEvent.matchId == matchId) & (dfEvent.playerId == playerId)]

        # dangerous stuff
        teamIn1H = dfEvent[dfEvent['matchId'] == matchId]['teamId'].iloc[0]
        right = attackToRight(dfEvent[dfEvent['matchId'] == matchId]['positions'].iloc[0])

        # for each event in the subset
        for inter in datasetToScroll.values:
            # if the event has 1401 (interception id)
            if (checkSpecificTagInsideList(inter[10], 1401)):
                counterPlayer = counterPlayer + 1
                # we want the accurate completed ones (check id 1801)
                if (checkSpecificTagInsideList(inter[10], 1801)):
                    counterCompletedInterceptions = counterCompletedInterceptions + 1
                    counterDangerousCompletedInterceptions = counterDangerousCompletedInterceptions + helperForUpgradingElement(
                        inter[11], matchId, inter[5], inter[7], right, True)
                    counterDangerousCompletedInterceptions = counterDangerousCompletedInterceptions + helperForUpgradingElement(
                        inter[11], matchId, inter[5], inter[7], right, False)

        listTotalInterceptions.append(counterPlayer)
        listCompletedInterceptions.append(counterCompletedInterceptions)
        listDangerousCompletedInterceptions.append(counterDangerousCompletedInterceptions)
        listDangerousId.append(playerId)
        listDangerousMatch.append(matchId)
    df['total_interceptions'] = listTotalInterceptions
    df['completed_interceptions'] = listCompletedInterceptions

    dangerousDF = pd.DataFrame({'id': listDangerousId,
                                'match': listDangerousMatch,
                                'dangerous_complete_interceptions': listDangerousCompletedInterceptions})

    return df, dangerousDF


def computeTotalNumberClearences(df, dfEvent):
    """
    For each player in df look the number of Clearances inside a match

    Parameters
    ----------
    df : match_id	gameweek	team_id	playerId	minutes_played	red_card	yellow_card	goals
        The dataset extracted with playerMatchExtraction from json file of Wyscout matches.

    dfEvent : The dataset that contain event of matches

    Returns
    -------
    df Dataframe
        The input data frame with the Clearances features included
    """
    listTotalClearences = []
    listCompletedClearance = []
    listFailedClearance = []
    listDangerousCompletedClearances = []
    listDangerousFailedClearances = []
    listDangerousId = []
    listDangerousMatch = []

    # Initial call to print 0% progress
    printProgressBar(0, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)

    # for each player in the dataset
    for index, playerMatch in df.iterrows():
        printProgressBar(index, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)
        # the number of air duel instantiated to 0 for each player
        counterPlayer = 0
        counterCompleteClearance = 0
        counterFailedClearance = 0
        dangerousFailedClearances = 0
        dangerousCompletedClearances = 0

        # we extract these information in order to reduce the size of the dataset to scroll
        matchId = playerMatch[0]
        playerId = playerMatch[3]
        datasetToScroll = dfEvent.loc[(dfEvent.matchId == matchId) & (dfEvent.playerId == playerId)]

        # dangerous stuff
        teamIn1H = dfEvent[dfEvent['matchId'] == matchId]['teamId'].iloc[0]
        right = attackToRight(dfEvent[dfEvent['matchId'] == matchId]['positions'].iloc[0])

        # for each event in the subset
        for clear in datasetToScroll.values:
            # if the event has 1501 (clearences id)
            if (checkSpecificTagInsideList(clear[10], 1501) or clear[9] == 'Clearance'):
                counterPlayer = counterPlayer + 1
                # we define accurate (1801 id)
                if (checkSpecificTagInsideList(clear[10], 1801)):
                    counterCompleteClearance = counterCompleteClearance + 1
                    dangerousCompletedClearances = dangerousCompletedClearances + helperForUpgradingElement(clear[11],
                                                                                                            matchId,
                                                                                                            clear[5],
                                                                                                            clear[7],
                                                                                                            right,
                                                                                                            False)
                # we define non accurate (1802 id)
                if (checkSpecificTagInsideList(clear[10], 1802)):
                    counterFailedClearance = counterFailedClearance + 1
                    dangerousFailedClearances = dangerousFailedClearances + helperForUpgradingElement(clear[11],
                                                                                                      matchId, clear[5],
                                                                                                      clear[7], right,
                                                                                                      False)

        listTotalClearences.append(counterPlayer)
        listCompletedClearance.append(counterCompleteClearance)
        listFailedClearance.append(counterFailedClearance)
        listDangerousId.append(playerId)
        listDangerousCompletedClearances.append(dangerousCompletedClearances)
        listDangerousFailedClearances.append(dangerousFailedClearances)
        listDangerousMatch.append(matchId)

    df['total_clearance'] = listTotalClearences
    df['completed_clearance'] = listCompletedClearance
    df['failed_clearance'] = listFailedClearance

    dangerousDF = pd.DataFrame({'id': listDangerousId,
                                'match': listDangerousMatch,
                                'dangerous_complete_clearances': listDangerousCompletedClearances,
                                'dangerous_failed_clearances': listDangerousFailedClearances})

    return df, dangerousDF


def computeTotalNumberShot(df, dfEvent):
    """
    For each player in df look the number of SHOT inside a match

    Parameters
    ----------
    df : match_id	gameweek	team_id	playerId	minutes_played	red_card	yellow_card	goals
        The dataset extracted with playerMatchExtraction from json file of Wyscout matches.

    dfEvent : The dataset that contain event of matches

    Returns
    -------
    df Dataframe
        The input data frame with the shot features included
    """
    listTotalShot = []
    listOffTargetShot = []
    listSavedShot = []
    listBlockedShot = []
    listGoal = []
    listWoodWork = []
    listOwnGoals = []
    listScoredPenalty = []
    listFailedPenalty = []
    size = len(df) - 1

    listDangerousOffTargetShot = []
    listDangerousSavedShot = []
    listDangerousBlockedShot = []
    listDangerousGoal = []
    listDangerousId = []
    listDangerousMatch = []

    # Initial call to print 0% progress
    printProgressBar(0, size, prefix='Progress:', suffix='Complete', length=50)

    # for each player in the dataset
    for bar, playerMatch in df.iterrows():
        printProgressBar(bar, size, prefix='Progress:', suffix='Complete', length=50)
        # the number of air duel instantiated to 0 for each player
        counterPlayer = 0
        counterOffTargetShot = 0
        counterSavedShot = 0
        counterBlockedShot = 0
        counterGoalShot = 0
        counterWoodWork = 0
        counterOwnGoal = 0
        counterScoredPenalty = 0
        counterFailedPenalty = 0

        dangerousOffTargetShot = 0
        dangerousSavedShot = 0
        dangerousBlockedShot = 0
        dangerousGoal = 0

        # we extract these information in order to reduce the size of the dataset to scroll
        matchId = playerMatch[0]
        playerId = playerMatch[3]
        datasetToScroll = dfEvent.loc[(dfEvent.matchId == matchId) & (dfEvent.playerId == playerId)]

        # dangerous stuff
        teamIn1H = dfEvent[dfEvent['matchId'] == matchId]['teamId'].iloc[0]
        right = attackToRight(dfEvent[dfEvent['matchId'] == matchId]['positions'].iloc[0])

        # for each event in the subset
        for shot in datasetToScroll.values:

            # we look for pass, more in deep crosses
            if (shot[1] == 'Shot'):
                counterPlayer = counterPlayer + 1
                # we look for out target shot
                # out id from 1210 to 1216
                if (checkSpecificTagInsideList(shot[10], 1210) or checkSpecificTagInsideList(shot[10], 1211) or
                        checkSpecificTagInsideList(shot[10], 1212) or checkSpecificTagInsideList(shot[10], 1213) or
                        checkSpecificTagInsideList(shot[10], 1214) or checkSpecificTagInsideList(shot[10], 1215) or
                        checkSpecificTagInsideList(shot[10], 1216)):
                    counterOffTargetShot = counterOffTargetShot + 1
                    dangerousOffTargetShot = dangerousOffTargetShot + helperForUpgradingElement(shot[11], matchId,
                                                                                                shot[5], shot[7], right,
                                                                                                True)

                # look for saved shot, so some shots that are inside the goal position but didn' produce goal id
                # not id 101 and all the id of goal postion (from 1201 to 1209)
                if (not (checkSpecificTagInsideList(shot[10], 101)) and (checkSpecificTagInsideList(shot[10], 1201) or
                                                                         checkSpecificTagInsideList(shot[10], 1202) or
                                                                         checkSpecificTagInsideList(shot[10], 1203) or
                                                                         checkSpecificTagInsideList(shot[10], 1204) or
                                                                         checkSpecificTagInsideList(shot[10], 1205) or
                                                                         checkSpecificTagInsideList(shot[10], 1206) or
                                                                         checkSpecificTagInsideList(shot[10], 1207) or
                                                                         checkSpecificTagInsideList(shot[10], 1208) or
                                                                         checkSpecificTagInsideList(shot[10], 1209))):
                    counterSavedShot = counterSavedShot + 1
                    dangerousSavedShot = dangerousSavedShot + helperForUpgradingElement(shot[11], matchId, shot[5],
                                                                                        shot[7], right, True)

                # we look for those shots that are blocked (id blocked shot 2101)
                if (checkSpecificTagInsideList(shot[10], 2101)):
                    counterBlockedShot = counterBlockedShot + 1
                    dangerousBlockedShot = dangerousBlockedShot + helperForUpgradingElement(shot[11], matchId, shot[5],
                                                                                            shot[7], right, True)

                # we look for goals (id 101)
                if (checkSpecificTagInsideList(shot[10], 101)):
                    counterGoalShot = counterGoalShot + 1
                    dangerousGoal = dangerousGoal + helperForUpgradingElement(shot[11], matchId, shot[5], shot[7],
                                                                              right, True)

                # we look for post target shot
                # out id from 1217 to 1223
                if (checkSpecificTagInsideList(shot[10], 1217) or checkSpecificTagInsideList(shot[10], 1218) or
                        checkSpecificTagInsideList(shot[10], 1219) or checkSpecificTagInsideList(shot[10], 1220) or
                        checkSpecificTagInsideList(shot[10], 1221) or checkSpecificTagInsideList(shot[10], 1222) or
                        checkSpecificTagInsideList(shot[10], 1223)):
                    counterWoodWork = counterWoodWork + 1

                # we look for own goals (id 102)
                if (checkSpecificTagInsideList(shot[10], 102)):
                    counterOwnGoal = counterOwnGoal + 1

            # we look also for free kick crosses
            if (shot[1] == 'Free Kick' and (
                    shot[9] == 'Free kick shot' or shot[9] == 'Goal kick' or shot[9] == 'Penalty')):
                counterPlayer = counterPlayer + 1
                # we look for out target shot
                # out id from 1210 to 1216
                if (checkSpecificTagInsideList(shot[10], 1210) or checkSpecificTagInsideList(shot[10], 1211) or
                        checkSpecificTagInsideList(shot[10], 1212) or checkSpecificTagInsideList(shot[10], 1213) or
                        checkSpecificTagInsideList(shot[10], 1214) or checkSpecificTagInsideList(shot[10], 1215) or
                        checkSpecificTagInsideList(shot[10], 1216)):
                    counterOffTargetShot = counterOffTargetShot + 1
                    dangerousOffTargetShot = dangerousOffTargetShot + helperForUpgradingElement(shot[11], matchId,
                                                                                                shot[5], shot[7], right,
                                                                                                True)
                    # off shot from penalty
                    if (shot[9] == 'Penalty'):
                        counterFailedPenalty = counterFailedPenalty + 1

                # look for saved shot, so some shots that are inside the goal position but didn' produce goal id
                # not id 101 and all the id of goal postion (from 1201 to 1209)
                if (not (checkSpecificTagInsideList(shot[10], 101)) and (checkSpecificTagInsideList(shot[10], 1201) or
                                                                         checkSpecificTagInsideList(shot[10], 1202) or
                                                                         checkSpecificTagInsideList(shot[10], 1203) or
                                                                         checkSpecificTagInsideList(shot[10], 1204) or
                                                                         checkSpecificTagInsideList(shot[10], 1205) or
                                                                         checkSpecificTagInsideList(shot[10], 1206) or
                                                                         checkSpecificTagInsideList(shot[10], 1207) or
                                                                         checkSpecificTagInsideList(shot[10], 1208) or
                                                                         checkSpecificTagInsideList(shot[10], 1209))):
                    counterSavedShot = counterSavedShot + 1
                    dangerousSavedShot = dangerousSavedShot + helperForUpgradingElement(shot[11], matchId, shot[5],
                                                                                        shot[7], right, True)

                    # saved from penalty
                    if (shot[9] == 'Penalty'):
                        counterFailedPenalty = counterFailedPenalty + 1

                # we look for those shots that are blocked (id blocked shot 2101)
                if (checkSpecificTagInsideList(shot[10], 2101)):
                    counterBlockedShot = counterBlockedShot + 1
                    dangerousBlockedShot = dangerousBlockedShot + helperForUpgradingElement(shot[11], matchId, shot[5],
                                                                                            shot[7], right, True)

                # we look for goals (id 101)
                if (checkSpecificTagInsideList(shot[10], 101)):
                    counterGoalShot = counterGoalShot + 1
                    dangerousGoal = dangerousGoal + helperForUpgradingElement(shot[11], matchId, shot[5], shot[7],
                                                                              right, True)

                    # if the goal came from a penalty
                    if (shot[9] == 'Penalty'):
                        counterScoredPenalty = counterScoredPenalty + 1

                        # we look for post target shot
                # out id from 1217 to 1223
                if (checkSpecificTagInsideList(shot[10], 1217) or checkSpecificTagInsideList(shot[10], 1218) or
                        checkSpecificTagInsideList(shot[10], 1219) or checkSpecificTagInsideList(shot[10], 1220) or
                        checkSpecificTagInsideList(shot[10], 1221) or checkSpecificTagInsideList(shot[10], 1222) or
                        checkSpecificTagInsideList(shot[10], 1223)):
                    counterWoodWork = counterWoodWork + 1
                    # wood work from penalty
                    if (shot[9] == 'Penalty'):
                        counterFailedPenalty = counterFailedPenalty + 1

                # we look for own goals (id 102)
                if (checkSpecificTagInsideList(shot[10], 102)):
                    counterOwnGoal = counterOwnGoal + 1

            # theere could be an own goal with a simple touch of the ball, this could be to an own goal
            # we look for subevent touch and we look for specific tag inside the list number 102 (own goal tag)
            if (shot[9] == 'Touch' and checkSpecificTagInsideList(shot[10], 102)):
                counterOwnGoal = counterOwnGoal + 1

        listTotalShot.append(counterPlayer)
        listBlockedShot.append(counterBlockedShot)
        listGoal.append(counterGoalShot)
        listOffTargetShot.append(counterOffTargetShot)
        listOwnGoals.append(counterOwnGoal)
        listSavedShot.append(counterSavedShot)
        listWoodWork.append(counterWoodWork)
        listFailedPenalty.append(counterFailedPenalty)
        listScoredPenalty.append(counterScoredPenalty)
        listDangerousOffTargetShot.append(dangerousOffTargetShot)
        listDangerousBlockedShot.append(dangerousBlockedShot)
        listDangerousSavedShot.append(dangerousSavedShot)
        listDangerousGoal.append(dangerousGoal)
        listDangerousId.append(playerId)
        listDangerousMatch.append(matchId)

    df['total_shot'] = listTotalShot
    df['blocked_shot'] = listBlockedShot
    df['goal_shot'] = listGoal
    df['off_target_shot'] = listOffTargetShot
    df['own_goals'] = listOwnGoals
    df['saved_shot'] = listSavedShot
    df['wood_worker_shot'] = listWoodWork
    df['failed_penalty'] = listFailedPenalty
    df['scored_penalty'] = listScoredPenalty

    dangerousDF = pd.DataFrame({'id': listDangerousId,
                                'match': listDangerousMatch,
                                'dangerous_off_target_shot': listDangerousOffTargetShot,
                                'dangerous_saved_shot': listDangerousSavedShot,
                                'dangeorus_blocked_shot': listDangerousBlockedShot,
                                'dangerous_goal_shot': listDangerousGoal})

    return df, dangerousDF


def populateBlameAndMerit(df, responsabilities, merit, saves):
    """
    For each match in the df create a match with the blame and responsabilties from function computeResponsabilitiesInPenalty

    Parameters
    ----------
    df : passed directely from the function computeResponsabilitiesInPenalty, is the same of the one used in the other function

    responsabilities : Dictionary structured as id match = { } and id_player = number of responsabilities about penalties in match

    Returns
    -------
    df Dataframe
        The input data frame with the analysis of penalties features included
    """
    arrayBlame = []
    arrayMerit = []
    arraySaves = []
    for record in df.values:
        # BLAME PART
        # we have a match with the match id
        if (record[0] in responsabilities):
            # there is a match with the playerid
            if (record[3] in responsabilities[record[0]]):
                arrayBlame.append(responsabilities[record[0]][record[3]])
            else:
                arrayBlame.append(0)
        else:
            arrayBlame.append(0)
        # MERIT PART
        # we have a match with the match id
        if (record[0] in merit):
            # there is a match with the player id
            if (record[3] in merit[record[0]]):
                arrayMerit.append(merit[record[0]][record[3]])
            else:
                arrayMerit.append(0)
        else:
            arrayMerit.append(0)
        # SAVES GOALKEEPER PART
        # we have a match with the match id
        if (record[0] in saves):
            # there is a match with the player id
            if (record[3] in saves[record[0]]):
                arraySaves.append(saves[record[0]][record[3]])
            else:
                arraySaves.append(0)
        else:
            arraySaves.append(0)
    df['blame_on_penalty'] = arrayBlame
    df['merit_on_penalty'] = arrayMerit
    df['penalty_saved'] = arraySaves

    return df


def computeResponsabilitiesInPenalty(df, listOfMatchesId, dfEvent):
    """
    For each match in the listOfMatchesId look the statistics regarding blame and merit in penalties in a game

    Parameters
    ----------
    df : match_id	gameweek	team_id	playerId	minutes_played	red_card	yellow_card	goals
        The dataset extracted with playerMatchExtraction from json file of Wyscout matches.

    dfEvent : The dataset that contain event of matches

    Returns
    -------
    df Dataframe
        The input data frame with the analysis of penalties features included
    """
    responsabilities = {}
    merit = {}
    saves = {}
    # for each matchid
    for match in listOfMatchesId:
        responsabilities[match] = {}
        # found a subset of dataset for a particulat match
        datasetToScroll = dfEvent.loc[(dfEvent.matchId == match)]
        # scroll all event inside the dataset to found penalty and the one that conquered it
        responsabilities[match] = {}
        merit[match] = {}
        saves[match] = {}
        foundGroundDefenseDuel = False
        foundGroundAttackDuel = False
        foundFirstGroundlooseballduel = False
        foundSecondGroundlooseballduel = False
        penaltyFound = False
        handFoul = False
        airduel1 = False
        airduel2 = False
        foul = False

        playerDefenseId = ''
        playerAttackId = ''
        playerDefenseTeam = ''
        playerAttackTeam = ''

        for event in datasetToScroll.values:
            # 4 options of penalty (attack vs defense) (air duel) (double Ground loose ball duel) (hand foul)

            # FIRST OPTION ground attaccking duel
            if (event[9] == 'Ground attacking duel'):
                playerAttackId = event[6]
                playerAttackTeam = event[11]
                foundGroundAttackDuel = True
                foundFirstGroundlooseballduel = False
                foundSecondGroundlooseballduel = False
                handFoul = False
                airduel1 = False
                airduel2 = False
                foul = False
                penaltyFound = False

            # defense ground duel
            elif (event[9] == 'Ground defending duel'):
                playerDefenseId = event[6]
                playerDefenseTeam = event[11]
                foundGroundDefenseDuel = True
                foundFirstGroundlooseballduel = False
                foundSecondGroundlooseballduel = False
                handFoul = False
                airduel1 = False
                airduel2 = False
                foul = False
                penaltyFound = False

            # SECOND OPTION found Ground loose ball duel
            elif (event[9] == 'Ground loose ball duel'):

                # if is the second
                if (foundFirstGroundlooseballduel):
                    playerDefenseId = event[6]
                    playerDefenseTeam = event[11]
                    # set true
                    foundSecondGroundlooseballduel = True
                    foundGroundDefenseDuel = False
                    foundGroundAttackDuel = False
                    handFoul = False
                    airduel1 = False
                    airduel2 = False
                    foul = False
                    penaltyFound = False

                else:
                    playerAttackId = event[6]
                    playerAttackTeam = event[11]
                    foundFirstGroundlooseballduel = True
                    foundGroundDefenseDuel = False
                    foundGroundAttackDuel = False
                    handFoul = False
                    airduel1 = False
                    airduel2 = False
                    foul = False
                    penaltyFound = False

            # THIRD OPTION double air duel
            elif (event[9] == 'Air duel'):
                if (airduel1):
                    playerAttackId = event[6]
                    playerAttackTeam = event[11]
                    airduel2 = True
                    foundGroundDefenseDuel = False
                    foundGroundAttackDuel = False
                    foundFirstGroundlooseballduel = False
                    foundSecondGroundlooseballduel = False
                    handFoul = False
                    foul = False
                    penaltyFound = False
                else:
                    playerDefenseId = event[6]
                    playerDefenseTeam = event[11]
                    airduel1 = True
                    foundGroundDefenseDuel = False
                    foundGroundAttackDuel = False
                    foundFirstGroundlooseballduel = False
                    foundSecondGroundlooseballduel = False
                    handFoul = False
                    foul = False
                    penaltyFound = False

            elif (event[9] == 'Hand foul'):
                playerDefenseId = event[6]
                playerDefenseTeam = event[11]
                handFoul = True
                foul = True
                foundGroundDefenseDuel = False
                foundGroundAttackDuel = False
                foundFirstGroundlooseballduel = False
                foundSecondGroundlooseballduel = False
                airduel1 = False
                airduel2 = False
                penaltyFound = False

            # core part
            elif (event[9] == 'Penalty'):
                penaltyFound = True
                # foul need to be true
                if (foul):
                    # list all possibilities
                    # FIRST ONE is the ground duel that lead to a foul and then to a penalty
                    if (foundGroundDefenseDuel and foundGroundAttackDuel):
                        # if we want the defense the team id of the player need to be different to the one of the penalty
                        if (playerDefenseTeam != event[11]):
                            # the player is already in responsabilities
                            if (playerDefenseId in responsabilities[match]):
                                responsabilities[match][playerDefenseId] = responsabilities[match][playerDefenseId] + 1
                            else:
                                responsabilities[match][playerDefenseId] = 1
                        else:
                            # the team is the same so is the attacker
                            # the player is already in merit
                            if (playerDefenseId in merit[match]):
                                merit[match][playerDefenseId] = merit[match][playerDefenseId] + 1
                            else:
                                merit[match][playerDefenseId] = 1

                        # if we want the defense the team id of the player need to be different to the one of the penalty
                        if (playerAttackTeam != event[11]):
                            # the player is already in responsabilities
                            if (playerAttackId in responsabilities[match]):
                                responsabilities[match][playerAttackId] = responsabilities[match][playerAttackId] + 1
                            else:
                                responsabilities[match][playerAttackId] = 1
                        else:
                            # the team is the same so is the attacker
                            # the player is already in merit
                            if (playerAttackTeam in merit[match]):
                                merit[match][playerAttackId] = merit[match][playerAttackId] + 1
                            else:
                                merit[match][playerAttackId] = 1
                    # SECOND RESPONSABILITIES is the ground duel
                    if (foundFirstGroundlooseballduel and foundSecondGroundlooseballduel):
                        # if we want the defense the team id of the player need to be different to the one of the penalty
                        if (playerDefenseTeam != event[11]):
                            # the player is already in responsabilities
                            if (playerDefenseId in responsabilities[match]):
                                responsabilities[match][playerDefenseId] = responsabilities[match][playerDefenseId] + 1
                            else:
                                responsabilities[match][playerDefenseId] = 1
                        else:
                            # the team is the same so is the attacker
                            # the player is already in merit
                            if (playerDefenseId in merit[match]):
                                merit[match][playerDefenseId] = merit[match][playerDefenseId] + 1
                            else:
                                merit[match][playerDefenseId] = 1

                        # if we want the defense the team id of the player need to be different to the one of the penalty
                        if (playerAttackTeam != event[11]):
                            # the player is already in responsabilities
                            if (playerAttackId in responsabilities[match]):
                                responsabilities[match][playerAttackId] = responsabilities[match][playerAttackId] + 1
                            else:
                                responsabilities[match][playerAttackId] = 1
                        else:
                            # the team is the same so is the attacker
                            # the player is already in merit
                            if (playerAttackTeam in merit[match]):
                                merit[match][playerAttackId] = merit[match][playerAttackId] + 1
                            else:
                                merit[match][playerAttackId] = 1
                    # THIRD OPTION DOUBLE AIR DUEL
                    if (airduel1 and airduel2):
                        # if we want the defense the team id of the player need to be different to the one of the penalty
                        if (playerDefenseTeam != event[11]):
                            # the player is already in responsabilities
                            if (playerDefenseId in responsabilities[match]):
                                responsabilities[match][playerDefenseId] = responsabilities[match][playerDefenseId] + 1
                            else:
                                responsabilities[match][playerDefenseId] = 1
                        else:
                            # the team is the same so is the attacker
                            # the player is already in merit
                            if (playerDefenseId in merit[match]):
                                merit[match][playerDefenseId] = merit[match][playerDefenseId] + 1
                            else:
                                merit[match][playerDefenseId] = 1

                        # if we want the defense the team id of the player need to be different to the one of the penalty
                        if (playerAttackTeam != event[11]):
                            # the player is already in responsabilities
                            if (playerAttackId in responsabilities[match]):
                                responsabilities[match][playerAttackId] = responsabilities[match][playerAttackId] + 1
                            else:
                                responsabilities[match][playerAttackId] = 1
                        else:
                            # the team is the same so is the attacker
                            # the player is already in merit
                            if (playerAttackTeam in merit[match]):
                                merit[match][playerAttackId] = merit[match][playerAttackId] + 1
                            else:
                                merit[match][playerAttackId] = 1
                    # FOURTH OPTION HAND FOUL
                    if (handFoul):
                        # we don't have to check nothing is player responsabilities
                        # the player is already in responsabilities
                        if (playerDefenseId in responsabilities[match]):
                            responsabilities[match][playerDefenseId] = responsabilities[match][playerDefenseId] + 1
                        else:
                            responsabilities[match][playerDefenseId] = 1

            elif (event[1] == 'Foul'):
                playerDefenseId = event[6]
                foul = True
                penaltyFound = False
            # we look also for saved goal
            elif (event[1] == 'Save attempt' and penaltyFound):
                # we look for non goal save attempt after penalty
                if (not (checkSpecificTagInsideList(event[10], 101))):
                    insert = event[6]
                    if (insert in saves[match]):
                        saves[match][insert] = saves[match][insert] + 1
                    else:
                        saves[match][insert] = 1

            else:
                # reset
                foundGroundDefenseDuel = False
                foundGroundAttackDuel = False
                foundFirstGroundlooseballduel = False
                foundSecondGroundlooseballduel = False
                handFoul = False
                airduel1 = False
                airduel2 = False
                foul = False
                playerDefenseId = ''
                playerAttackId = ''
                playerDefenseTeam = ''
                playerAttackTeam = ''
                penaltyFound = False
    # we call the function that for each match in dictionary assign an array value for merit and blame to each player
    df = populateBlameAndMerit(df, responsabilities, merit, saves)

    return df


def computeTotalNumberGoalKeeperSaves(df, dfEvent):
    """
    For each player in df look the number of GOALKEEPER SAVES inside a match

    Parameters
    ----------
    df : match_id	gameweek	team_id	playerId	minutes_played	red_card	yellow_card	goals
        The dataset extracted with playerMatchExtraction from json file of Wyscout matches.

    dfEvent : The dataset that contain event of matches

    Returns
    -------
    df Dataframe
        The input data frame with the goalkeeper saves features included
    """
    listTotalGKSaves = []
    listTotalCatches = []
    listTotalPunches = []
    listFailedCaches = []
    listSufferedGoal = []
    # Initial call to print 0% progress
    printProgressBar(0, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)
    # for each player in the dataset
    for index, playerMatch in df.iterrows():
        printProgressBar(index, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)
        # the number of air duel instantiated to 0 for each player
        counterPlayer = 0
        counterCleanCatches = 0
        counterPunches = 0
        counterFailedCaches = 0
        counterGoalTaken = 0
        # we extract these information in order to reduce the size of the dataset to scroll
        matchId = playerMatch[0]
        playerId = playerMatch[3]
        datasetToScroll = dfEvent.loc[(dfEvent.matchId == matchId) & (dfEvent.playerId == playerId)]
        # for each event in the subset
        for saves in datasetToScroll.values:
            # if the event has 1501 (clearences id)
            if (saves[1] == 'Save attempt' and (saves[9] == 'Save attempt' or saves[9] == 'Reflexes')):
                # no goal is a save
                if (not (checkSpecificTagInsideList(saves[10], 101))):
                    counterPlayer = counterPlayer + 1
                # we look for accurate save attempt for cach tag 1801 for catch????
                if (checkSpecificTagInsideList(saves[10], 1801)):
                    counterCleanCatches = counterCleanCatches + 1
                # punches as goalkeeper (we want inaccurate catches) 1802 for innacurate so punches??:
                if (checkSpecificTagInsideList(saves[10], 1802) and not (checkSpecificTagInsideList(saves[10], 101))):
                    counterPunches = counterPunches + 1
                # we look for failed chaches accurate one that lead to goal???
                if (checkSpecificTagInsideList(saves[10], 1801) and checkSpecificTagInsideList(saves[10], 101)):
                    counterFailedCaches = counterFailedCaches + 1

                if (checkSpecificTagInsideList(saves[10], 101)):
                    counterGoalTaken = counterGoalTaken + 1

        listTotalGKSaves.append(counterPlayer)
        listTotalCatches.append(counterCleanCatches)
        listTotalPunches.append(counterPunches)
        listFailedCaches.append(counterFailedCaches)
        listSufferedGoal.append(counterGoalTaken)
    df['save_goal_keeping'] = listTotalGKSaves
    df['catch_goal_keeping'] = listTotalCatches
    df['punch_goal_keeping'] = listTotalPunches
    df['failed_catch_goal_keeping'] = listFailedCaches
    df['suffered_goal'] = listSufferedGoal

    return df


def computeTotalNumberFoul(df, dfEvent):
    """
    For each player in df look the number of foul inside a match

    Parameters
    ----------
    df : match_id	gameweek	team_id	playerId	minutes_played	red_card	yellow_card	goals
        The dataset extracted with playerMatchExtraction from json file of Wyscout matches.

    dfEvent : The dataset that contain event of matches

    Returns
    -------
    df Dataframe
        The input data frame with the foul features included
    """
    listTotalFouls = []
    listYellowCard = []
    listRedCard = []
    listSecondYellow = []
    listDangerousFoul = []
    listDangerousId = []
    listDangerousMatch = []

    # Initial call to print 0% progress
    printProgressBar(0, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)

    # for each player in the dataset
    for index, playerMatch in df.iterrows():
        printProgressBar(index, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)
        # the number of air duel instantiated to 0 for each player
        counterPlayer = 0
        counterYellow = 0
        counterRed = 0
        counter2Yellow = 0
        dangerousFoul = 0

        # we extract these information in order to reduce the size of the dataset to scroll
        matchId = playerMatch[0]
        playerId = playerMatch[3]
        datasetToScroll = dfEvent.loc[(dfEvent.matchId == matchId) & (dfEvent.playerId == playerId)]

        # dangerous stuff
        teamIn1H = dfEvent[dfEvent['matchId'] == matchId]['teamId'].iloc[0]
        right = attackToRight(dfEvent[dfEvent['matchId'] == matchId]['positions'].iloc[0])

        # for each event in the subset
        for foul in datasetToScroll.values:
            if (foul[1] == 'Foul'):
                counterPlayer = counterPlayer + 1
                dangerousFoul = dangerousFoul + helperForUpgradingElement(foul[11], matchId, foul[5], foul[7], right,
                                                                          False)
                # id tag 1702 yellow card
                if (checkSpecificTagInsideList(foul[10], 1702)):
                    counterYellow = counterYellow + 1
                # id tag 1703 second yellow card
                if (checkSpecificTagInsideList(foul[10], 1703)):
                    counter2Yellow = counter2Yellow + 1
                # id tag 1701 red card
                if (checkSpecificTagInsideList(foul[10], 1701)):
                    counterRed = counterRed + 1
        listTotalFouls.append(counterPlayer)
        listYellowCard.append(counterYellow)
        listRedCard.append(counterRed)
        listSecondYellow.append(counter2Yellow)
        listDangerousFoul.append(dangerousFoul)
        listDangerousId.append(playerId)
        listDangerousMatch.append(matchId)

    df['foul_made'] = listTotalFouls
    df['yellow_card_features_extarction'] = listYellowCard
    df['red_card_features_extarction'] = listRedCard
    df['second_yellow_card_features_extarction'] = listSecondYellow

    dangerousDF = pd.DataFrame({'id': listDangerousId,
                                'match': listDangerousMatch,
                                'dangerous_number_foul': listDangerousFoul})

    return df, dangerousDF


def populateSufferedFoul(df, responsabilities, merit, saves):
    """
    For each match in the df create a match with the number of suffered faul from function computeFoulFeature

    Parameters
    ----------
    df : passed directely from the function computeResponsabilitiesInPenalty, is the same of the one used in the other function

    responsabilities : Dictionary structured as id match = { } and id_player = number of suffered foul in match

    Returns
    -------
    df Dataframe
        The input data frame with the analysis of foul features included
    """
    arrayBlame = []
    arrayMerit = []
    arraySaves = []
    for record in df.values:
        # BLAME PART
        # we have a match with the match id
        if (record[0] in responsabilities):
            # there is a match with the playerid
            if (record[3] in responsabilities[record[0]]):
                arrayBlame.append(responsabilities[record[0]][record[3]])
            else:
                arrayBlame.append(0)
        else:
            arrayBlame.append(0)
        # MERIT PART
        # we have a match with the match id
        if (record[0] in merit):
            # there is a match with the player id
            if (record[3] in merit[record[0]]):
                arrayMerit.append(merit[record[0]][record[3]])
            else:
                arrayMerit.append(0)
        else:
            arrayMerit.append(0)
        # SAVES GOALKEEPER PART
        # we have a match with the match id
        if (record[0] in saves):
            # there is a match with the player id
            if (record[3] in saves[record[0]]):
                arraySaves.append(saves[record[0]][record[3]])
            else:
                arraySaves.append(0)
        else:
            arraySaves.append(0)
    df['suffered_foul'] = arrayBlame
    df['tackle_foul'] = arrayMerit
    df['headed_foul'] = arraySaves

    return df


def populateDangersSufferedFoul(df, suffered, tackle, head, matches, players):
    """
    For each match in the df create a match with the number of suffered faul from function computeFoulFeature

    Parameters
    ----------
    df : passed directely from the function computeResponsabilitiesInPenalty

    responsabilities : Dictionary structured as id_player = number of suffered foul in match

    Returns
    -------
    df Dataframe
        The input data frame with the analysis of foul features included
    """
    arraySuffered = []
    arrayTackle = []
    arrayHead = []
    arrayId = []
    arrayMatch = []

    # retrive match id information
    for match in matches:
        # retrive player
        listOfPlayers = players[players['match_id'] == match]['playerId'].unique()

        for player in listOfPlayers:
            foundSuffered = 0
            foundTackle = 0
            foundHead = 0

            # for each player id and value in suffered
            for player1, numbers in suffered[match].items():
                if (player == player1):
                    foundSuffered = numbers
            for playerTackle, numbersTackle in tackle[match].items():
                # same player
                if (player1 == playerTackle):
                    foundTackle = numbersTackle
            for playerHead, numbersHead in head[match].items():
                if (player1 == playerHead):
                    foundHead = numbersHead
            arraySuffered.append(foundSuffered)
            arrayTackle.append(foundTackle)
            arrayHead.append(foundHead)
            arrayMatch.append(match)
            arrayId.append(player)

    df['id'] = arrayId
    df['match'] = arrayMatch
    df['dangerous_foul_suffered'] = arraySuffered
    df['dangerous_tackle_foul'] = arrayTackle
    df['dangerous_head_foul'] = arrayHead

    return df


def computeFoulFeatures(df, listUniqueMatchId, dfEvent):
    """
    For each match in the listUniqueMatcheId look the statistics regarding Fauls in a game

    Parameters
    ----------
    df : match_id	gameweek	team_id	playerId	minutes_played	red_card	yellow_card	goals
        The dataset extracted with playerMatchExtraction from json file of Wyscout matches.

    dfEvent : The dataset that contain event of matches

    Returns
    -------
    df Dataframe
        The input data frame with the analysis of faul features included
    """
    suffered = {}
    head = {}
    tackle = {}

    dangerousSuffered = {}
    dangerousHead = {}
    dangeorusTackle = {}

    # for each match
    for match in listUniqueMatchId:
        suffered[match] = {}
        head[match] = {}
        tackle[match] = {}

        dangerousSuffered[match] = {}
        dangerousHead[match] = {}
        dangeorusTackle[match] = {}

        # found a subset of dataset for a particulat match
        datasetToScroll = dfEvent.loc[(dfEvent.matchId == match)]

        # dangerous stuff
        teamIn1H = dfEvent[dfEvent['matchId'] == match]['teamId'].iloc[0]
        right = attackToRight(dfEvent[dfEvent['matchId'] == match]['positions'].iloc[0])

        # scroll all event inside the dataset to found penalty and the one that conquered it
        foundGroundDefenseDuel = False
        foundGroundAttackDuel = False
        tackles = False
        foundFirstGroundlooseballduel = False
        foundSecondGroundlooseballduel = False
        penaltyFound = False
        handFoul = False
        airduel1 = False
        airduel2 = False
        foul = False

        playerDefenseId = ''
        playerAttackId = ''
        playerDefenseTeam = ''
        playerAttackTeam = ''

        for event in datasetToScroll.values:

            if (event[6] in dangerousSuffered[match]):
                p = 0
            else:
                dangerousSuffered[match][event[6]] = 0
            if (event[6] in dangeorusTackle[match]):
                p = 0
            else:
                dangeorusTackle[match][event[6]] = 0
            if (event[6] in dangerousHead[match]):
                p = 0
            else:
                dangerousHead[match][event[6]] = 0

            # 4 options of penalty (attack vs defense) (air duel) (double Ground loose ball duel) (hand foul)

            # FIRST OPTION ground attaccking duel
            if (event[9] == 'Ground attacking duel'):
                playerAttackId = event[6]
                playerAttackTeam = event[11]
                foundGroundAttackDuel = True
                foundFirstGroundlooseballduel = False
                foundSecondGroundlooseballduel = False
                handFoul = False
                airduel1 = False
                airduel2 = False
                foul = False
                penaltyFound = False

            # defense ground duel
            elif (event[9] == 'Ground defending duel'):
                playerDefenseId = event[6]
                playerDefenseTeam = event[11]
                if (checkSpecificTagInsideList(event[10], 1601)):
                    tackles = True
                foundGroundDefenseDuel = True
                foundFirstGroundlooseballduel = False
                foundSecondGroundlooseballduel = False
                handFoul = False
                airduel1 = False
                airduel2 = False
                foul = False
                penaltyFound = False

            # SECOND OPTION found Ground loose ball duel
            elif (event[9] == 'Ground loose ball duel'):

                # if is the second
                if (foundFirstGroundlooseballduel):
                    playerDefenseId = event[6]
                    playerDefenseTeam = event[11]
                    if (checkSpecificTagInsideList(event[10], 1601)):
                        tackles = True
                    # set true
                    foundSecondGroundlooseballduel = True
                    foundGroundDefenseDuel = False
                    foundGroundAttackDuel = False
                    handFoul = False
                    airduel1 = False
                    airduel2 = False
                    foul = False
                    penaltyFound = False

                else:
                    playerAttackId = event[6]
                    playerAttackTeam = event[11]
                    if (checkSpecificTagInsideList(event[10], 1601)):
                        tackles = True
                    foundFirstGroundlooseballduel = True
                    foundGroundDefenseDuel = False
                    foundGroundAttackDuel = False
                    handFoul = False
                    airduel1 = False
                    airduel2 = False
                    foul = False
                    penaltyFound = False

            # THIRD OPTION double air duel
            elif (event[9] == 'Air duel'):
                if (airduel1):
                    playerAttackId = event[6]
                    playerAttackTeam = event[11]
                    airduel2 = True
                    foundGroundDefenseDuel = False
                    foundGroundAttackDuel = False
                    foundFirstGroundlooseballduel = False
                    foundSecondGroundlooseballduel = False
                    handFoul = False
                    tackles = False
                    foul = False
                    penaltyFound = False
                else:
                    playerDefenseId = event[6]
                    playerDefenseTeam = event[11]
                    airduel1 = True
                    foundGroundDefenseDuel = False
                    tackles = False
                    foundGroundAttackDuel = False
                    foundFirstGroundlooseballduel = False
                    foundSecondGroundlooseballduel = False
                    handFoul = False
                    foul = False
                    penaltyFound = False

            elif (event[9] == 'Hand foul'):
                playerDefenseId = event[6]
                playerDefenseTeam = event[11]
                handFoul = True
                foul = True
                tackles = False
                foundGroundDefenseDuel = False
                foundGroundAttackDuel = False
                foundFirstGroundlooseballduel = False
                foundSecondGroundlooseballduel = False
                airduel1 = False
                airduel2 = False
                penaltyFound = False

            elif (event[1] == 'Foul'):
                playerDefenseId = event[6]
                # FIRST ONE is the ground duel that lead to a foul and then to a penalty
                if (foundGroundDefenseDuel and foundGroundAttackDuel):
                    # if we want the defense the team id of the player need to be different to the one of the faul
                    if (playerDefenseTeam != event[11]):
                        # the player is already in suffered
                        if (playerDefenseId in suffered[match]):
                            suffered[match][playerDefenseId] = suffered[match][playerDefenseId] + 1
                        else:
                            suffered[match][playerDefenseId] = 1

                            # DANGEROUS the player is already in suffered
                        if (playerDefenseId in dangerousSuffered[match]):
                            dangerousSuffered[match][playerDefenseId] = dangerousSuffered[match][
                                                                            playerDefenseId] + helperForUpgradingElement(
                                event[11], match, event[5], event[7], right, True)
                        else:
                            dangerousSuffered[match][playerDefenseId] = helperForUpgradingElement(event[11], match,
                                                                                                  event[5], event[7],
                                                                                                  right, True)
                    else:
                        # the player made a foul
                        # since is on land we need to check if is a tackle
                        if (tackles):
                            if (playerDefenseId in tackle[match]):
                                tackle[match][playerDefenseId] = tackle[match][playerDefenseId] + 1
                            else:
                                tackle[match][playerDefenseId] = 1

                            # DANGEROUS the player is already in suffered
                            if (playerDefenseId in dangeorusTackle[match]):
                                dangeorusTackle[match][playerDefenseId] = dangeorusTackle[match][
                                                                              playerDefenseId] + helperForUpgradingElement(
                                    event[11], match, event[5], event[7], right, False)
                            else:
                                dangeorusTackle[match][playerDefenseId] = helperForUpgradingElement(event[11], match,
                                                                                                    event[5], event[7],
                                                                                                    right, False)

                    # if we want the defense the team id of the player need to be different to the one of the faul
                    if (playerAttackTeam != event[11]):
                        # the player is already in responsabilities
                        if (playerAttackId in suffered[match]):
                            suffered[match][playerAttackId] = suffered[match][playerAttackId] + 1
                        else:
                            suffered[match][playerAttackId] = 1
                        # DANGEROUS the player is already in suffered
                        if (playerAttackId in dangerousSuffered[match]):
                            dangerousSuffered[match][playerAttackId] = dangerousSuffered[match][
                                                                           playerAttackId] + helperForUpgradingElement(
                                event[11], match, event[5], event[7], right, True)
                        else:
                            dangerousSuffered[match][playerAttackId] = helperForUpgradingElement(event[11], match,
                                                                                                 event[5], event[7],
                                                                                                 right, True)

                    else:
                        # the player made a foul
                        # since is on land we need to check if is a tackle
                        if (tackles):
                            if (playerDefenseId in tackle[match]):
                                tackle[match][playerDefenseId] = tackle[match][playerDefenseId] + 1
                            else:
                                tackle[match][playerDefenseId] = 1

                            # DANGEROUS the player is already in suffered
                            if (playerDefenseId in dangeorusTackle[match]):
                                dangeorusTackle[match][playerDefenseId] = dangeorusTackle[match][
                                                                              playerDefenseId] + helperForUpgradingElement(
                                    event[11], match, event[5], event[7], right, False)
                            else:
                                dangeorusTackle[match][playerDefenseId] = helperForUpgradingElement(event[11], match,
                                                                                                    event[5], event[7],
                                                                                                    right, False)

                # SECOND RESPONSABILITIES is the ground duel
                if (foundFirstGroundlooseballduel and foundSecondGroundlooseballduel):
                    # if we want the defense the team id of the player need to be different to the one of the penalty
                    if (playerDefenseTeam != event[11]):
                        # the player is already in responsabilities
                        if (playerDefenseId in suffered[match]):
                            suffered[match][playerDefenseId] = suffered[match][playerDefenseId] + 1
                        else:
                            suffered[match][playerDefenseId] = 1

                        # DANGEROUS the player is already in suffered
                        if (playerDefenseId in dangerousSuffered[match]):
                            dangerousSuffered[match][playerDefenseId] = dangerousSuffered[match][
                                                                            playerDefenseId] + helperForUpgradingElement(
                                event[11], match, event[5], event[7], right, True)
                        else:
                            dangerousSuffered[match][playerDefenseId] = helperForUpgradingElement(event[11], match,
                                                                                                  event[5], event[7],
                                                                                                  right, True)

                    else:
                        # the player made a foul
                        # since is on land we need to check if is a tackle
                        if (tackles):
                            if (playerDefenseId in tackle[match]):
                                tackle[match][playerDefenseId] = tackle[match][playerDefenseId] + 1
                            else:
                                tackle[match][playerDefenseId] = 1

                            # DANGEROUS the player is already in suffered
                            if (playerDefenseId in dangeorusTackle[match]):
                                dangeorusTackle[match][playerDefenseId] = dangeorusTackle[match][
                                                                              playerDefenseId] + helperForUpgradingElement(
                                    event[11], match, event[5], event[7], right, False)
                            else:
                                dangeorusTackle[match][playerDefenseId] = helperForUpgradingElement(event[11], match,
                                                                                                    event[5], event[7],
                                                                                                    right, False)

                    # if we want the defense the team id of the player need to be different to the one of the penalty
                    if (playerAttackTeam != event[11]):
                        # the player is already in responsabilities
                        if (playerAttackId in suffered[match]):
                            suffered[match][playerAttackId] = suffered[match][playerAttackId] + 1
                        else:
                            suffered[match][playerAttackId] = 1
                        # DANGEROUS the player is already in suffered
                        if (playerAttackId in dangerousSuffered[match]):
                            dangerousSuffered[match][playerAttackId] = dangerousSuffered[match][
                                                                           playerAttackId] + helperForUpgradingElement(
                                event[11], match, event[5], event[7], right, True)
                        else:
                            dangerousSuffered[match][playerAttackId] = helperForUpgradingElement(event[11], match,
                                                                                                 event[5], event[7],
                                                                                                 right, True)
                    else:
                        # the player made a foul
                        # since is on land we need to check if is a tackle
                        if (tackles):
                            if (playerDefenseId in tackle[match]):
                                tackle[match][playerDefenseId] = tackle[match][playerDefenseId] + 1
                            else:
                                tackle[match][playerDefenseId] = 1
                            # DANGEROUS the player is already in suffered
                            if (playerDefenseId in dangeorusTackle[match]):
                                dangeorusTackle[match][playerDefenseId] = dangeorusTackle[match][
                                                                              playerDefenseId] + helperForUpgradingElement(
                                    event[11], match, event[5], event[7], right, False)
                            else:
                                dangeorusTackle[match][playerDefenseId] = helperForUpgradingElement(event[11], match,
                                                                                                    event[5], event[7],
                                                                                                    right, False)

                # THIRD OPTION DOUBLE AIR DUEL
                if (airduel1 and airduel2):
                    # if we want the defense the team id of the player need to be different to the one of the faul
                    if (playerDefenseTeam != event[11]):
                        # the player is already in responsabilities
                        if (playerDefenseId in suffered[match]):
                            suffered[match][playerDefenseId] = suffered[match][playerDefenseId] + 1
                        else:
                            suffered[match][playerDefenseId] = 1

                        # DANGEROUS the player is already in suffered
                        if (playerDefenseId in dangerousSuffered[match]):
                            dangerousSuffered[match][playerDefenseId] = dangerousSuffered[match][
                                                                            playerDefenseId] + helperForUpgradingElement(
                                event[11], match, event[5], event[7], right, True)
                        else:
                            dangerousSuffered[match][playerDefenseId] = helperForUpgradingElement(event[11], match,
                                                                                                  event[5], event[7],
                                                                                                  right, True)

                    else:
                        # the team is the same so is the defensor
                        # the player is already in head
                        if (playerDefenseId in head[match]):
                            head[match][playerDefenseId] = head[match][playerDefenseId] + 1
                        else:
                            head[match][playerDefenseId] = 1
                        # DANGEROUS the player is already in suffered
                        if (playerDefenseId in dangerousHead[match]):
                            dangerousHead[match][playerDefenseId] = dangerousHead[match][
                                                                        playerDefenseId] + helperForUpgradingElement(
                                event[11], match, event[5], event[7], right, False)
                        else:
                            dangerousHead[match][playerDefenseId] = helperForUpgradingElement(event[11], match,
                                                                                              event[5], event[7], right,
                                                                                              False)

                    # if we want the defense the team id of the player need to be different to the one of the foul
                    if (playerAttackTeam != event[11]):
                        # the player is already in responsabilities
                        if (playerAttackId in suffered[match]):
                            suffered[match][playerAttackId] = suffered[match][playerAttackId] + 1
                        else:
                            suffered[match][playerAttackId] = 1
                        # DANGEROUS the player is already in suffered
                        if (playerAttackId in dangerousSuffered[match]):
                            dangerousSuffered[match][playerAttackId] = dangerousSuffered[match][
                                                                           playerAttackId] + helperForUpgradingElement(
                                event[11], match, event[5], event[7], right, True)
                        else:
                            dangerousSuffered[match][playerAttackId] = helperForUpgradingElement(event[11], match,
                                                                                                 event[5], event[7],
                                                                                                 right, True)

                    else:
                        # the team is the same so is the defenser
                        if (playerDefenseId in head[match]):
                            head[match][playerDefenseId] = head[match][playerDefenseId] + 1
                        else:
                            head[match][playerDefenseId] = 1
                            # DANGEROUS the player is already in suffered
                        if (playerDefenseId in dangerousHead[match]):
                            dangerousHead[match][playerDefenseId] = dangerousHead[match][
                                                                        playerDefenseId] + helperForUpgradingElement(
                                event[11], match, event[5], event[7], right, False)
                        else:
                            dangerousHead[match][playerDefenseId] = helperForUpgradingElement(event[11], match,
                                                                                              event[5], event[7], right,
                                                                                              False)

            else:
                # reset
                foundGroundDefenseDuel = False
                foundGroundAttackDuel = False
                foundFirstGroundlooseballduel = False
                foundSecondGroundlooseballduel = False
                handFoul = False
                airduel1 = False
                airduel2 = False
                tackles = False
                foul = False
                playerDefenseId = ''
                playerAttackId = ''
                playerDefenseTeam = ''
                playerAttackTeam = ''
                penaltyFound = False
    # we call the function that for each match in dictionary assign an array value for merit and blame to each player
    df = populateSufferedFoul(df, suffered, head, tackle)
    dangerousDF = pd.DataFrame()
    dangerousDF = populateDangersSufferedFoul(dangerousDF, dangerousSuffered, dangeorusTackle, dangerousHead,
                                              df['match_id'].unique(), df)

    return df, dangerousDF


def computeTotalActionPartecipation(df, dfListMatchId, dfEvent):
    """
    For each match in the dfListMatchId look the statistics regarding action participation in a game

    Parameters
    ----------
    df : match_id	gameweek	team_id	playerId	minutes_played	red_card	yellow_card	goals
        The dataset extracted with playerMatchExtraction from json file of Wyscout matches.

    dfEvent : The dataset that contain event of matches

    Returns
    -------
    df Dataframe
        The input data frame with the action partecipation features included
    """
    # parametro per il controllo della lunghezza dell'azione
    n = 1
    dictionaryNumbersOfActions = {}
    dictionaryNumbersOfActionsToShot = {}
    # GOAL PART
    dictionaryNumbersOfActionsToGoal = {}
    # for each single match id
    for match in dfListMatchId:
        dictionaryNumbersOfActions[match] = {}
        dictionaryNumbersOfActionsToShot[match] = {}
        # GOAL PART
        dictionaryNumbersOfActionsToGoal[match] = {}
        # create a sub dataframe to scroll in order to define a timeline for the event
        dfToScroll = dfEvent.loc[(dfEvent.matchId == match)]
        listOfPlayerIdInAction = []
        teamId = 0
        first = True
        # for each event
        for event in dfToScroll.values:
            # print(str(event[1]) + " Of the team " + str(event[11]))
            if (first):
                # assign team
                teamId = event[11]
                listOfPlayerIdInAction.append(event[6])
                first = False
            else:
                # there is a change in team
                if (teamId != event[11]):
                    # the new team may lost the duel
                    if (checkSpecificTagInsideList(event[10], 701) or checkSpecificTagInsideList(event[10], 702)):
                        # maybe an air duel or ground loose.
                        pippo = ''
                    else:
                        # there is a new action, or interruption is done. The action is over.
                        # change team
                        teamId = event[11]
                        # if the len is more than n we add the ids to
                        if (len(listOfPlayerIdInAction) > n):
                            dictionaryNumbersOfActions = fromArrayToDict(listOfPlayerIdInAction,
                                                                         dictionaryNumbersOfActions, match)
                        listOfPlayerIdInAction = []
                        # print('CAMBIO POSSESSO O INTERRUZIONE AZIONE')
                else:
                    if (event[1] == 'Shot'):
                        listOfPlayerIdInAction.append(event[6])
                        # implementation for completed action
                        dictionaryNumbersOfActionsToShot = fromArrayToDict(listOfPlayerIdInAction,
                                                                           dictionaryNumbersOfActionsToShot, match)
                    # GOAL PART goal at the end of the action
                    elif (checkSpecificTagInsideList(event[10], 101)):
                        listOfPlayerIdInAction.append(event[6])
                        # implementation for completed action
                        dictionaryNumbersOfActionsToGoal = fromArrayToDict(listOfPlayerIdInAction,
                                                                           dictionaryNumbersOfActionsToGoal, match)
                    else:
                        listOfPlayerIdInAction.append(event[6])

    df = populateDataframeWithActionPartecipation(df, dictionaryNumbersOfActions, 'total_action_partecipation')
    df = populateDataframeWithActionPartecipation(df, dictionaryNumbersOfActionsToShot,
                                                  'successful_action_partecipation')
    # GOAL PART
    df = populateDataframeWithActionPartecipation(df, dictionaryNumbersOfActionsToGoal, 'action_partecipation_to_goal')

    return df


def fromArrayToDict(playersAction, dictionaryAction, match):
    for el in playersAction:
        if (el in dictionaryAction[match]):
            dictionaryAction[match][el] = dictionaryAction[match][el] + 1
        else:
            dictionaryAction[match][el] = 1
    return dictionaryAction


def populateDataframeWithActionPartecipation(df, dictionaryOfPartecipation, fieldName):
    arrayPartecipation = []
    for record in df.values:
        # Partecipation part
        # we have a match with the match id
        if (record[0] in dictionaryOfPartecipation):
            # there is a match with the playerid
            if (record[3] in dictionaryOfPartecipation[record[0]]):
                arrayPartecipation.append(dictionaryOfPartecipation[record[0]][record[3]])
            else:
                arrayPartecipation.append(0)
        else:
            arrayPartecipation.append(0)
    df[fieldName] = arrayPartecipation
    return df


def qualityFeaturesComputationManager(dfToFill, dfevent):
    print(len(dfToFill))
    print('Computing Passes Features....')
    dfToFill, toMergeDangerousPasses = computeTotalPasses(dfToFill, dfevent)
    print(len(dfToFill))
    print('Computing Cross Features....')
    dfToFill, toMergeDangerousCross = computeTotalNumberCross(dfToFill, dfevent)
    print(len(dfToFill))
    print('Computing Tackels Features....')
    dfToFill, toMergeDangerousTackels = computeTotalNumberTackels(dfToFill, dfevent)
    print(len(dfToFill))
    print('Computing Dribbling Features....')
    dfToFill, toMergeDangerousDribblings = computeTotalNumberDribbling(dfToFill, dfevent)
    print(len(dfToFill))
    print('Computing Air duel Features....')
    dfToFill, toMergeDangerousAirDuel = computeTotalAirDuel(dfToFill, dfevent)
    print(len(dfToFill))
    print('Computing Interceptions Features....')
    dfToFill, toMergeDangerousInterceptions = computeTotalNumberInterceptions(dfToFill, dfevent)
    print(len(dfToFill))
    print('Computing Clearences Features....')
    dfToFill, toMergeDangerousClearances = computeTotalNumberClearences(dfToFill, dfevent)
    print(len(dfToFill))
    print('Computing Shot Features....')
    dfToFill, toMergeDangerousShot = computeTotalNumberShot(dfToFill, dfevent)
    print(len(dfToFill))
    print('Computing Blame, Merit and Saves in Penalty Features....')
    dfToFill = computeResponsabilitiesInPenalty(dfToFill, dfToFill['match_id'].unique(), dfevent)
    print(len(dfToFill))
    print('Computing GoalKeeper Features....')
    dfToFill = computeTotalNumberGoalKeeperSaves(dfToFill, dfevent)
    print(len(dfToFill))
    print('Computing Foul Features....')
    dfToFill, toMergeDangerousFaulHT = computeFoulFeatures(dfToFill, dfToFill['match_id'].unique(), dfevent)
    print(len(dfToFill))
    dfToFill, toMergeDangerousFoul = computeTotalNumberFoul(dfToFill, dfevent)
    print(len(dfToFill))
    print('Computing Action Partecipation Features')
    dfToFill = computeTotalActionPartecipation(dfToFill, dfToFill['match_id'].unique(), dfevent)
    print(len(dfToFill))

    print('Merge Phase..')
    df_merge_col = pd.merge(toMergeDangerousPasses, toMergeDangerousCross, on=['id', 'match'])
    print(len(df_merge_col))
    df_merge_col = pd.merge(df_merge_col, toMergeDangerousTackels, on=['id', 'match'])
    print(len(df_merge_col))
    df_merge_col = pd.merge(df_merge_col, toMergeDangerousDribblings, on=['id', 'match'])
    print(len(df_merge_col))
    df_merge_col = pd.merge(df_merge_col, toMergeDangerousAirDuel, on=['id', 'match'])
    print(len(df_merge_col))
    df_merge_col = pd.merge(df_merge_col, toMergeDangerousInterceptions, on=['id', 'match'])
    print(len(df_merge_col))
    df_merge_col = pd.merge(df_merge_col, toMergeDangerousClearances, on=['id', 'match'])
    print(len(df_merge_col))
    df_merge_col = pd.merge(df_merge_col, toMergeDangerousShot, on=['id', 'match'])
    print(len(df_merge_col))
    df_merge_col = pd.merge(df_merge_col, toMergeDangerousFaulHT, on=['id', 'match'])
    print(len(df_merge_col))
    df_merge_col = pd.merge(df_merge_col, toMergeDangerousFoul, on=['id', 'match'])
    print(len(df_merge_col))

    print('---------------------------------------------------------------------')
    print('Conclusion Report:')
    print('Lenght df to fill at the end : ' + str(len(dfToFill)))
    print('Lenght dangerous Feature Merge : ' + str(len(df_merge_col)))

    return dfToFill, df_merge_col


def computeTotalForEachMatchTeam(dfToFill):
    '''
    For each quality feature, for each match for each team compute the total quantity regarding quality features
    ------
    Params
    dfToFill: Datframe with the features extract in the same order as the on passed in this example by input

                Index(['match_id', 'gameweek', 'team_id', 'playerId', 'minutes_played',
       'red_card', 'yellow_card', 'goals', 'total_passes', 'completed_passes',
       'failed_passes', 'key_passes', 'assist_passes', 'total_cross',
       'completed_cross', 'failed_cross', 'assist_cross', 'total_tackels',
       'completed_tackels', 'failed_tackels', 'total_dribblings',
       'completed_dribblings', 'failed_dribblings', 'headed_duel',
       'complete_headed_duel', 'failed_headed_duel', 'total_interceptions',
       'completed_interceptions', 'total_clearance', 'completed_clearance',
       'failed_clearance', 'total_shot', 'blocked_shot', 'goal_shot',
       'off_target_shot', 'own_goals', 'saved_shot', 'wood_worker_shot',
       'failed_penalty', 'scored_penalty', 'blame_on_penalty',
       'merit_on_penalty', 'penalty_saved', 'save_goal_keeping',
       'catch_goal_keeping', 'punch_goal_keeping', 'failed_catch_goal_keeping',
       'suffered_goal', 'suffered_foul', 'tackle_foul', 'headed_foul',
       'foul_made', 'yellow_card_features_extarction',
       'red_card_features_extarction',
       'second_yellow_card_features_extarction', 'total_action_partecipation',
       'successful_action_partecipation'])
    Return
    DF: new dataframe with the record stored as match id, team id , total quantity for each quality feature

    '''
    # take the list of all matches id
    listOfMatchId = dfToFill['match_id'].unique()
    arrayMatch = []
    arrayTeam = []
    arrayTotalPasses = []
    arrayCompletedPasses = []
    arrayFailedPasses = []
    arrayKeyPasses = []
    arratAssistPasses = []
    arrayTotalCross = []
    arrayCompletedCross = []
    arrayFailedCross = []
    arrayAssistCross = []
    arrayTotalTackels = []
    arrayCompletedTackels = []
    arrayFailedTackels = []
    arrayTotalDribblings = []
    arrayCompletedDribblings = []
    arrayFailedDribblings = []
    arrayHeadedDuel = []
    arrayCompleteHeadedDuel = []
    arrayFailedHeadedDuel = []
    arrayTotalInterceptions = []
    arrayCompletedInterceptions = []
    arrayTotalClearance = []
    arrayCompletedClearance = []
    arrayFailedClearance = []
    arrayTotalShot = []
    arrayBlockedShot = []
    arrayGoalShot = []
    arrayOffTargetShot = []
    arrayOwnGoals = []
    arraySavedShot = []
    arrayWoodWorkerShot = []
    arrayFailedPenalty = []
    arrayScoredPenalty = []
    arrayBlameOnPenalty = []
    arrayMeritOnPenalty = []
    arrayPenaltySaved = []
    arraySaveGoal = []
    arrayCaychGoal = []
    arrayPunchGoal = []
    arrayFailedCatchGoal = []
    arraySufferedGoal = []
    arraySufferedFoul = []
    arrayTackleFoul = []
    arrayHeadedFoul = []
    arrayFoulMade = []
    arrayYellowCard = []
    arrayRedCard = []
    arraySecondYellow = []
    arrayTotalAction = []
    arrayTotalActionShot = []

    print('Starting Grand Total Value Computation for Quality Features')
    from tqdm import tqdm
    with tqdm(total=len(listOfMatchId)) as pbar:
        # for each match, we look into dfToFill and we extract the total count for each team
        for match in listOfMatchId:
            pbar.update(1)
            # extract the team id for each game
            listOfTeamId = dfToFill[dfToFill['match_id'] == match]['team_id'].unique()
            # for each team we pass through the sub dataset of match team
            for team in listOfTeamId:
                TotalPasses = 0
                CompletedPasses = 0
                FailedPasses = 0
                KeyPasses = 0
                AssistPasses = 0
                TotalCross = 0
                CompletedCross = 0
                FailedCross = 0
                AssistCross = 0
                TotalTackels = 0
                CompletedTackels = 0
                FailedTackels = 0
                TotalDribblings = 0
                CompletedDribblings = 0
                FailedDribblings = 0
                HeadedDuel = 0
                CompleteHeadedDuel = 0
                FailedHeadedDuel = 0
                TotalInterceptions = 0
                CompletedInterceptions = 0
                TotalClearance = 0
                CompletedClearance = 0
                FailedClearance = 0
                TotalShot = 0
                BlockedShot = 0
                GoalShot = 0
                OffTargetShot = 0
                OwnGoals = 0
                SavedShot = 0
                WoodWorkerShot = 0
                FailedPenalty = 0
                ScoredPenalty = 0
                BlameOnPenalty = 0
                MeritOnPenalty = 0
                PenaltySaved = 0
                SaveGoal = 0
                CaychGoal = 0
                PunchGoal = 0
                FailedCatchGoal = 0
                SufferedGoal = 0
                SufferedFoul = 0
                TackleFoul = 0
                HeadedFoul = 0
                FoulMade = 0
                YellowCard = 0
                RedCard = 0
                SecondYellow = 0
                TotalAction = 0
                TotalActionShot = 0
                # extract the subdataset with team and match id
                subDataset = dfToFill.loc[(dfToFill.match_id == match) & (dfToFill.team_id == team)]
                # for each player collect the qualitative features in order to have the total sum
                for player in subDataset.values:
                    TotalPasses = TotalPasses + player[8]
                    CompletedPasses = CompletedPasses + player[9]
                    FailedPasses = FailedPasses + player[10]
                    KeyPasses = KeyPasses + player[11]
                    AssistPasses = AssistPasses + player[12]
                    TotalCross = TotalCross + player[13]
                    CompletedCross = CompletedCross + player[14]
                    FailedCross = FailedCross + player[15]
                    AssistCross = AssistCross + player[16]
                    TotalTackels = TotalTackels + player[17]
                    CompletedTackels = CompletedTackels + player[18]
                    FailedTackels = FailedTackels + player[19]
                    TotalDribblings = TotalDribblings + player[20]
                    CompletedDribblings = CompletedDribblings + player[21]
                    FailedDribblings = FailedDribblings + player[22]
                    HeadedDuel = HeadedDuel + player[23]
                    CompleteHeadedDuel = CompleteHeadedDuel + player[24]
                    FailedHeadedDuel = FailedHeadedDuel + player[25]
                    TotalInterceptions = TotalInterceptions + player[26]
                    CompletedInterceptions = CompletedInterceptions + player[27]
                    TotalClearance = TotalClearance + player[28]
                    CompletedClearance = CompletedClearance + player[29]
                    FailedClearance = FailedClearance + player[30]
                    TotalShot = TotalShot + player[31]
                    BlockedShot = BlockedShot + player[32]
                    GoalShot = GoalShot + player[33]
                    OffTargetShot = OffTargetShot + player[34]
                    OwnGoals = OwnGoals + player[35]
                    SavedShot = SavedShot + player[36]
                    WoodWorkerShot = WoodWorkerShot + player[37]
                    FailedPenalty = FailedPenalty + player[38]
                    ScoredPenalty = ScoredPenalty + player[39]
                    BlameOnPenalty = BlameOnPenalty + player[40]
                    MeritOnPenalty = MeritOnPenalty + player[41]
                    PenaltySaved = PenaltySaved + player[42]
                    SaveGoal = SaveGoal + player[43]
                    CaychGoal = CaychGoal + player[44]
                    PunchGoal = PunchGoal + player[45]
                    FailedCatchGoal = FailedCatchGoal + player[46]
                    SufferedGoal = SufferedGoal + player[47]
                    SufferedFoul = SufferedFoul + player[48]
                    TackleFoul = TackleFoul + player[49]
                    HeadedFoul = HeadedFoul + player[50]
                    FoulMade = FoulMade + player[51]
                    YellowCard = YellowCard + player[52]
                    RedCard = RedCard + player[53]
                    SecondYellow = SecondYellow + player[54]
                    TotalAction = TotalAction + player[55]
                    TotalActionShot = TotalActionShot + player[56]
                arrayMatch.append(match)
                arrayTeam.append(team)
                arrayTotalPasses.append(TotalPasses)
                arrayCompletedPasses.append(CompletedPasses)
                arrayFailedPasses.append(FailedPasses)
                arrayKeyPasses.append(KeyPasses)
                arratAssistPasses.append(AssistPasses)
                arrayTotalCross.append(TotalCross)
                arrayCompletedCross.append(CompletedCross)
                arrayFailedCross.append(FailedCross)
                arrayAssistCross.append(AssistCross)
                arrayTotalTackels.append(TotalTackels)
                arrayCompletedTackels.append(CompletedTackels)
                arrayFailedTackels.append(FailedTackels)
                arrayTotalDribblings.append(TotalDribblings)
                arrayCompletedDribblings.append(CompletedDribblings)
                arrayFailedDribblings.append(FailedDribblings)
                arrayHeadedDuel.append(HeadedDuel)
                arrayCompleteHeadedDuel.append(CompleteHeadedDuel)
                arrayFailedHeadedDuel.append(FailedHeadedDuel)
                arrayTotalInterceptions.append(TotalInterceptions)
                arrayCompletedInterceptions.append(CompletedInterceptions)
                arrayTotalClearance.append(TotalClearance)
                arrayCompletedClearance.append(CompletedClearance)
                arrayFailedClearance.append(FailedClearance)
                arrayTotalShot.append(TotalShot)
                arrayBlockedShot.append(BlockedShot)
                arrayGoalShot.append(GoalShot)
                arrayOffTargetShot.append(OffTargetShot)
                arrayOwnGoals.append(OwnGoals)
                arraySavedShot.append(SavedShot)
                arrayWoodWorkerShot.append(WoodWorkerShot)
                arrayFailedPenalty.append(FailedPenalty)
                arrayScoredPenalty.append(ScoredPenalty)
                arrayBlameOnPenalty.append(BlameOnPenalty)
                arrayMeritOnPenalty.append(MeritOnPenalty)
                arrayPenaltySaved.append(PenaltySaved)
                arraySaveGoal.append(SaveGoal)
                arrayCaychGoal.append(CaychGoal)
                arrayPunchGoal.append(PunchGoal)
                arrayFailedCatchGoal.append(FailedCatchGoal)
                arraySufferedGoal.append(SufferedGoal)
                arraySufferedFoul.append(SufferedFoul)
                arrayTackleFoul.append(TackleFoul)
                arrayHeadedFoul.append(HeadedFoul)
                arrayFoulMade.append(FoulMade)
                arrayYellowCard.append(YellowCard)
                arrayRedCard.append(RedCard)
                arraySecondYellow.append(SecondYellow)
                arrayTotalAction.append(TotalAction)
                arrayTotalActionShot.append(TotalActionShot)
    totalQuality = pd.DataFrame({'match_id_contribution_feature': arrayMatch,
                                 'team_id_contribution_feature': arrayTeam,
                                 'total_passes_contribution_feature': arrayTotalPasses,
                                 'completed_passes_contribution_feature': arrayCompletedPasses,
                                 'failed_passes_contribution_feature': arrayFailedPasses,
                                 'key_passes_contribution_feature': arrayKeyPasses,
                                 'assist_passes_contribution_feature': arratAssistPasses,
                                 'total_cross_contribution_feature': arrayTotalCross,
                                 'completed_cross_contribution_feature': arrayCompletedCross,
                                 'failed_cross_contribution_feature': arrayFailedCross,
                                 'assist_cross_contribution_feature': arrayAssistCross,
                                 'total_tackels_contribution_feature': arrayTotalTackels,
                                 'completed_tackels_contribution_feature': arrayCompletedTackels,
                                 'failed_tackels_contribution_feature': arrayFailedTackels,
                                 'total_dribblings_contribution_feature': arrayTotalDribblings,
                                 'completed_dribblings_contribution_feature': arrayCompletedDribblings,
                                 'failed_dribblings_contribution_feature': arrayFailedDribblings,
                                 'headed_duel_contribution_feature': arrayHeadedDuel,
                                 'complete_headed_duel_contribution_feature': arrayCompleteHeadedDuel,
                                 'failed_headed_duel_contribution_feature': arrayFailedHeadedDuel,
                                 'total_interceptions_contribution_feature': arrayTotalInterceptions,
                                 'completed_interceptions_contribution_feature': arrayCompletedInterceptions,
                                 'total_clearance_contribution_feature': arrayTotalClearance,
                                 'completed_clearance_contribution_feature': arrayCompletedClearance,
                                 'failed_clearance_contribution_feature': arrayFailedClearance,
                                 'total_shot_contribution_feature': arrayTotalShot,
                                 'blocked_shot_contribution_feature': arrayBlockedShot,
                                 'goal_shot_contribution_feature': arrayGoalShot,
                                 'off_target_shot_contribution_feature': arrayOffTargetShot,
                                 'own_goals_contribution_feature': arrayOwnGoals,
                                 'saved_shot_contribution_feature': arraySavedShot,
                                 'wood_worker_shot_contribution_feature': arrayWoodWorkerShot,
                                 'failed_penalty_contribution_feature': arrayFailedPenalty,
                                 'scored_penalty_contribution_feature': arrayScoredPenalty,
                                 'blame_on_penalty_contribution_feature': arrayBlameOnPenalty,
                                 'merit_on_penalty_contribution_feature': arrayMeritOnPenalty,
                                 'penalty_saved_contribution_feature': arrayPenaltySaved,
                                 'save_goal_keeping_contribution_feature': arraySaveGoal,
                                 'catch_goal_keeping_contribution_feature': arrayCaychGoal,
                                 'punch_goal_keeping_contribution_feature': arrayPunchGoal,
                                 'failed_catch_goal_keeping_contribution_feature': arrayFailedCatchGoal,
                                 'suffered_goal_contribution_feature': arraySufferedGoal,
                                 'suffered_foul_contribution_feature': arraySufferedFoul,
                                 'tackle_foul_contribution_feature': arrayTackleFoul,
                                 'headed_foul_contribution_feature': arrayHeadedFoul,
                                 'foul_made_contribution_feature': arrayFoulMade,
                                 'yellow_card_features_extarction_contribution_feature': arrayYellowCard,
                                 'red_card_features_extarction_contribution_feature': arrayRedCard,
                                 'second_yellow_card_features_extarction_contribution_feature': arraySecondYellow,
                                 'total_action_partecipation_contribution_feature': arrayTotalAction,
                                 'successful_action_partecipation_contribution_feature': arrayTotalActionShot})
    return totalQuality

def weird_division(n, d):
    return n / d if d else 0


def computeContributionFeatures(df, dfWithTotal):
    '''
    For each quality feature, for each match for each team for each player compute the contribution feature regarding quality features
    ------
    Params
    dfToFill: Datframe with the features extract in the same order as the on passed in this example by input

                Index(['match_id', 'gameweek', 'team_id', 'playerId', 'minutes_played',
       'red_card', 'yellow_card', 'goals', 'total_passes', 'completed_passes',
       'failed_passes', 'key_passes', 'assist_passes', 'total_cross',
       'completed_cross', 'failed_cross', 'assist_cross', 'total_tackels',
       'completed_tackels', 'failed_tackels', 'total_dribblings',
       'completed_dribblings', 'failed_dribblings', 'headed_duel',
       'complete_headed_duel', 'failed_headed_duel', 'total_interceptions',
       'completed_interceptions', 'total_clearance', 'completed_clearance',
       'failed_clearance', 'total_shot', 'blocked_shot', 'goal_shot',
       'off_target_shot', 'own_goals', 'saved_shot', 'wood_worker_shot',
       'failed_penalty', 'scored_penalty', 'blame_on_penalty',
       'merit_on_penalty', 'penalty_saved', 'save_goal_keeping',
       'catch_goal_keeping', 'punch_goal_keeping', 'failed_catch_goal_keeping',
       'suffered_goal', 'suffered_foul', 'tackle_foul', 'headed_foul',
       'foul_made', 'yellow_card_features_extarction',
       'red_card_features_extarction',
       'second_yellow_card_features_extarction', 'total_action_partecipation',
       'successful_action_partecipation'])

    dfWithTotal: Dataframe with the grand total for each team for each player
            Example:
            Index(['match_id_contribution_feature', 'team_id_contribution_feature',
                   'total_passes_contribution_feature',
                   'completed_passes_contribution_feature',
                   'failed_passes_contribution_feature', 'key_passes_contribution_feature',
                   'assist_passes_contribution_feature',
                   'total_cross_contribution_feature',
                   'completed_cross_contribution_feature',
                   'failed_cross_contribution_feature',
                   'assist_cross_contribution_feature',
                   'total_tackels_contribution_feature',
                   'completed_tackels_contribution_feature',
                   'failed_tackels_contribution_feature',
                   'total_dribblings_contribution_feature',
                   'completed_dribblings_contribution_feature',
                   'failed_dribblings_contribution_feature',
                   'headed_duel_contribution_feature',
                   'complete_headed_duel_contribution_feature',
                   'failed_headed_duel_contribution_feature',
                   'total_interceptions_contribution_feature',
                   'completed_interceptions_contribution_feature',
                   'total_clearance_contribution_feature',
                   'completed_clearance_contribution_feature',
                   'failed_clearance_contribution_feature',
                   'total_shot_contribution_feature', 'blocked_shot_contribution_feature',
                   'goal_shot_contribution_feature',
                   'off_target_shot_contribution_feature',
                   'own_goals_contribution_feature', 'saved_shot_contribution_feature',
                   'wood_worker_shot_contribution_feature',
                   'failed_penalty_contribution_feature',
                   'scored_penalty_contribution_feature',
                   'blame_on_penalty_contribution_feature',
                   'merit_on_penalty_contribution_feature',
                   'penalty_saved_contribution_feature',
                   'save_goal_keeping_contribution_feature',
                   'catch_goal_keeping_contribution_feature',
                   'punch_goal_keeping_contribution_feature',
                   'failed_catch_goal_keeping_contribution_feature',
                   'suffered_goal_contribution_feature',
                   'suffered_foul_contribution_feature',
                   'tackle_foul_contribution_feature', 'headed_foul_contribution_feature',
                   'foul_made_contribution_feature',
                   'yellow_card_features_extarction_contribution_feature',
                   'red_card_features_extarction_contribution_feature',
                   'second_yellow_card_features_extarction_contribution_feature',
                   'total_action_partecipation_contribution_feature',
                   'successful_action_partecipation_contribution_feature'],
                  dtype='object')

    Return
    DF: dataframe with the contribution features Extracted

    '''
    # instanziate all the arrays needed
    arrayTotalPasses = []
    arrayCompletedPasses = []
    arrayFailedPasses = []
    arrayKeyPasses = []
    arratAssistPasses = []
    arrayTotalCross = []
    arrayCompletedCross = []
    arrayFailedCross = []
    arrayAssistCross = []
    arrayTotalTackels = []
    arrayCompletedTackels = []
    arrayFailedTackels = []
    arrayTotalDribblings = []
    arrayCompletedDribblings = []
    arrayFailedDribblings = []
    arrayHeadedDuel = []
    arrayCompleteHeadedDuel = []
    arrayFailedHeadedDuel = []
    arrayTotalInterceptions = []
    arrayCompletedInterceptions = []
    arrayTotalClearance = []
    arrayCompletedClearance = []
    arrayFailedClearance = []
    arrayTotalShot = []
    arrayBlockedShot = []
    arrayGoalShot = []
    arrayOffTargetShot = []
    arrayOwnGoals = []
    arraySavedShot = []
    arrayWoodWorkerShot = []
    arrayFailedPenalty = []
    arrayScoredPenalty = []
    arrayBlameOnPenalty = []
    arrayMeritOnPenalty = []
    arrayPenaltySaved = []
    arraySaveGoal = []
    arrayCaychGoal = []
    arrayPunchGoal = []
    arrayFailedCatchGoal = []
    arraySufferedGoal = []
    arraySufferedFoul = []
    arrayTackleFoul = []
    arrayHeadedFoul = []
    arrayFoulMade = []
    arrayYellowCard = []
    arrayRedCard = []
    arraySecondYellow = []
    arrayTotalAction = []
    arrayTotalActionShot = []
    print('Starting Single Player Contributive Features Computation...')
    with tqdm(total=len(df)) as pbar:
        # for each player match team
        for player in df.values:
            pbar.update(1)
            TotalPasses = 0
            CompletedPasses = 0
            FailedPasses = 0
            KeyPasses = 0
            AssistPasses = 0
            TotalCross = 0
            CompletedCross = 0
            FailedCross = 0
            AssistCross = 0
            TotalTackels = 0
            CompletedTackels = 0
            FailedTackels = 0
            TotalDribblings = 0
            CompletedDribblings = 0
            FailedDribblings = 0
            HeadedDuel = 0
            CompleteHeadedDuel = 0
            FailedHeadedDuel = 0
            TotalInterceptions = 0
            CompletedInterceptions = 0
            TotalClearance = 0
            CompletedClearance = 0
            FailedClearance = 0
            TotalShot = 0
            BlockedShot = 0
            GoalShot = 0
            OffTargetShot = 0
            OwnGoals = 0
            SavedShot = 0
            WoodWorkerShot = 0
            FailedPenalty = 0
            ScoredPenalty = 0
            BlameOnPenalty = 0
            MeritOnPenalty = 0
            PenaltySaved = 0
            SaveGoal = 0
            CaychGoal = 0
            PunchGoal = 0
            FailedCatchGoal = 0
            SufferedGoal = 0
            SufferedFoul = 0
            TackleFoul = 0
            HeadedFoul = 0
            FoulMade = 0
            YellowCard = 0
            RedCard = 0
            SecondYellow = 0
            TotalAction = 0
            TotalActionShot = 0
            # for each team match
            for teamMatch in dfWithTotal.values:
                # check if the match and the team are the same
                if (player[0] == teamMatch[0] and player[2] == teamMatch[1]):
                    # print('Divido ' + str(player[8]) + ' con ' + str(teamMatch[2]) + '   risultato = ' + str(player[8]/teamMatch[2]) )
                    # compute metrics
                    TotalPasses = weird_division(player[8], teamMatch[2])
                    CompletedPasses = weird_division(player[9], teamMatch[3])
                    FailedPasses = weird_division(player[10], teamMatch[4])
                    KeyPasses = weird_division(player[11], teamMatch[5])
                    AssistPasses = weird_division(player[12], teamMatch[6])
                    TotalCross = weird_division(player[13], teamMatch[7])
                    CompletedCross = weird_division(player[14], teamMatch[8])
                    FailedCross = weird_division(player[15], teamMatch[9])
                    AssistCross = weird_division(player[16], teamMatch[10])
                    TotalTackels = weird_division(player[17], teamMatch[11])
                    CompletedTackels = weird_division(player[18], teamMatch[12])
                    FailedTackels = weird_division(player[19], teamMatch[13])
                    TotalDribblings = weird_division(player[20], teamMatch[14])
                    CompletedDribblings = weird_division(player[21], teamMatch[15])
                    FailedDribblings = weird_division(player[22], teamMatch[16])
                    HeadedDuel = weird_division(player[23], teamMatch[17])
                    CompleteHeadedDuel = weird_division(player[24], teamMatch[18])
                    FailedHeadedDuel = weird_division(player[25], teamMatch[19])
                    TotalInterceptions = weird_division(player[26], teamMatch[20])
                    CompletedInterceptions = weird_division(player[27], teamMatch[21])
                    TotalClearance = weird_division(player[28], teamMatch[22])
                    CompletedClearance = weird_division(player[29], teamMatch[23])
                    FailedClearance = weird_division(player[30], teamMatch[24])
                    TotalShot = weird_division(player[31], teamMatch[25])
                    BlockedShot = weird_division(player[32], teamMatch[26])
                    GoalShot = weird_division(player[33], teamMatch[27])
                    OffTargetShot = weird_division(player[34], teamMatch[28])
                    OwnGoals = weird_division(player[35], teamMatch[29])
                    SavedShot = weird_division(player[36], teamMatch[30])
                    WoodWorkerShot = weird_division(player[37], teamMatch[31])
                    FailedPenalty = weird_division(player[38], teamMatch[32])
                    ScoredPenalty = weird_division(player[39], teamMatch[33])
                    BlameOnPenalty = weird_division(player[40], teamMatch[34])
                    MeritOnPenalty = weird_division(player[41], teamMatch[35])
                    PenaltySaved = weird_division(player[42], teamMatch[36])
                    SaveGoal = weird_division(player[43], teamMatch[37])
                    CaychGoal = weird_division(player[44], teamMatch[38])
                    PunchGoal = weird_division(player[45], teamMatch[39])
                    FailedCatchGoal = weird_division(player[46], teamMatch[40])
                    SufferedGoal = weird_division(player[47], teamMatch[41])
                    SufferedFoul = weird_division(player[48], teamMatch[42])
                    TackleFoul = weird_division(player[49], teamMatch[43])
                    HeadedFoul = weird_division(player[50], teamMatch[44])
                    FoulMade = weird_division(player[51], teamMatch[45])
                    YellowCard = weird_division(player[52], teamMatch[46])
                    RedCard = weird_division(player[53], teamMatch[47])
                    SecondYellow = weird_division(player[54], teamMatch[48])
                    TotalAction = weird_division(player[55], teamMatch[49])
                    TotalActionShot = weird_division(player[56], teamMatch[50])
            # append metrics in arrays
            arrayTotalPasses.append(TotalPasses)
            arrayCompletedPasses.append(CompletedPasses)
            arrayFailedPasses.append(FailedPasses)
            arrayKeyPasses.append(KeyPasses)
            arratAssistPasses.append(AssistPasses)
            arrayTotalCross.append(TotalCross)
            arrayCompletedCross.append(CompletedCross)
            arrayFailedCross.append(FailedCross)
            arrayAssistCross.append(AssistCross)
            arrayTotalTackels.append(TotalTackels)
            arrayCompletedTackels.append(CompletedTackels)
            arrayFailedTackels.append(FailedTackels)
            arrayTotalDribblings.append(TotalDribblings)
            arrayCompletedDribblings.append(CompletedDribblings)
            arrayFailedDribblings.append(FailedDribblings)
            arrayHeadedDuel.append(HeadedDuel)
            arrayCompleteHeadedDuel.append(CompleteHeadedDuel)
            arrayFailedHeadedDuel.append(FailedHeadedDuel)
            arrayTotalInterceptions.append(TotalInterceptions)
            arrayCompletedInterceptions.append(CompletedInterceptions)
            arrayTotalClearance.append(TotalClearance)
            arrayCompletedClearance.append(CompletedClearance)
            arrayFailedClearance.append(FailedClearance)
            arrayTotalShot.append(TotalShot)
            arrayBlockedShot.append(BlockedShot)
            arrayGoalShot.append(GoalShot)
            arrayOffTargetShot.append(OffTargetShot)
            arrayOwnGoals.append(OwnGoals)
            arraySavedShot.append(SavedShot)
            arrayWoodWorkerShot.append(WoodWorkerShot)
            arrayFailedPenalty.append(FailedPenalty)
            arrayScoredPenalty.append(ScoredPenalty)
            arrayBlameOnPenalty.append(BlameOnPenalty)
            arrayMeritOnPenalty.append(MeritOnPenalty)
            arrayPenaltySaved.append(PenaltySaved)
            arraySaveGoal.append(SaveGoal)
            arrayCaychGoal.append(CaychGoal)
            arrayPunchGoal.append(PunchGoal)
            arrayFailedCatchGoal.append(FailedCatchGoal)
            arraySufferedGoal.append(SufferedGoal)
            arraySufferedFoul.append(SufferedFoul)
            arrayTackleFoul.append(TackleFoul)
            arrayHeadedFoul.append(HeadedFoul)
            arrayFoulMade.append(FoulMade)
            arrayYellowCard.append(YellowCard)
            arrayRedCard.append(RedCard)
            arraySecondYellow.append(SecondYellow)
            arrayTotalAction.append(TotalAction)
            arrayTotalActionShot.append(TotalActionShot)

    df['total_passes_contribution_feature'] = arrayTotalPasses
    df['completed_passes_contribution_feature'] = arrayCompletedPasses
    df['failed_passes_contribution_feature'] = arrayFailedPasses
    df['key_passes_contribution_feature'] = arrayKeyPasses
    df['assist_passes_contribution_feature'] = arratAssistPasses
    df['total_cross_contribution_feature'] = arrayTotalCross
    df['completed_cross_contribution_feature'] = arrayCompletedCross
    df['failed_cross_contribution_feature'] = arrayFailedCross
    df['assist_cross_contribution_feature'] = arrayAssistCross
    df['total_tackels_contribution_feature'] = arrayTotalTackels
    df['completed_tackels_contribution_feature'] = arrayCompletedTackels
    df['failed_tackels_contribution_feature'] = arrayFailedTackels
    df['total_dribblings_contribution_feature'] = arrayTotalDribblings
    df['completed_dribblings_contribution_feature'] = arrayCompletedDribblings
    df['failed_dribblings_contribution_feature'] = arrayFailedDribblings
    df['headed_duel_contribution_feature'] = arrayHeadedDuel
    df['complete_headed_duel_contribution_feature'] = arrayCompleteHeadedDuel
    df['failed_headed_duel_contribution_feature'] = arrayFailedHeadedDuel
    df['total_interceptions_contribution_feature'] = arrayTotalInterceptions
    df['completed_interceptions_contribution_feature'] = arrayCompletedInterceptions
    df['total_clearance_contribution_feature'] = arrayTotalClearance
    df['completed_clearance_contribution_feature'] = arrayCompletedClearance
    df['failed_clearance_contribution_feature'] = arrayFailedClearance
    df['total_shot_contribution_feature'] = arrayTotalShot
    df['blocked_shot_contribution_feature'] = arrayBlockedShot
    df['goal_shot_contribution_feature'] = arrayGoalShot
    df['off_target_shot_contribution_feature'] = arrayOffTargetShot
    df['own_goals_contribution_feature'] = arrayOwnGoals
    df['saved_shot_contribution_feature'] = arraySavedShot
    df['wood_worker_shot_contribution_feature'] = arrayWoodWorkerShot
    df['failed_penalty_contribution_feature'] = arrayFailedPenalty
    df['scored_penalty_contribution_feature'] = arrayScoredPenalty
    df['blame_on_penalty_contribution_feature'] = arrayBlameOnPenalty
    df['merit_on_penalty_contribution_feature'] = arrayMeritOnPenalty
    df['penalty_saved_contribution_feature'] = arrayPenaltySaved
    df['save_goal_keeping_contribution_feature'] = arraySaveGoal
    df['catch_goal_keeping_contribution_feature'] = arrayCaychGoal
    df['punch_goal_keeping_contribution_feature'] = arrayPunchGoal
    df['failed_catch_goal_keeping_contribution_feature'] = arrayFailedCatchGoal
    df['suffered_goal_contribution_feature'] = arraySufferedGoal
    df['suffered_foul_contribution_feature'] = arraySufferedFoul
    df['tackle_foul_contribution_feature'] = arrayTackleFoul
    df['headed_foul_contribution_feature'] = arrayHeadedFoul
    df['foul_made_contribution_feature'] = arrayFoulMade
    df['yellow_card_features_extarction_contribution_feature'] = arrayYellowCard
    df['red_card_features_extarction_contribution_feature'] = arrayRedCard
    df['second_yellow_card_features_extarction_contribution_feature'] = arraySecondYellow
    df['total_action_partecipation_contribution_feature'] = arrayTotalAction
    df['successful_action_partecipation_contribution_feature'] = arrayTotalActionShot
    df['yellow_card_features_extarction_contribution_feature'] = arrayYellowCard
    df['red_card_features_extarction_contribution_feature'] = arrayRedCard
    df['second_yellow_card_features_extarction_contribution_feature'] = arraySecondYellow
    df['total_action_partecipation_contribution_feature'] = arrayTotalAction
    df['successful_action_partecipation_contribution_feature'] = arrayTotalActionShot

    return df


import math


def meanPassesRecoil(dfToFill, dfEvent):
    '''
    This function compute the mean lenght of passes in a game for each player

    Params:
        df that contains all the information regarding player and game week and other, so a dataset
            that at this point in computation count more or less 130 columns
        df event the json dataset that store all the games event

    Return:
        df that contains 131 columns with the mean Passes recoil feature added
    '''
    listMeanPassesRecoil = []
    listMeanPassesRecoilTeam = []
    listMeanPassesRecoilMatch = []

    print('Computing Individual Pass Mean Lenght...')
    # Initial call to print 0% progress
    printProgressBar(0, len(dfToFill) - 1, prefix='Progress:', suffix='Complete', length=50)
    # for each player
    for index, player in dfToFill.iterrows():
        printProgressBar(index, len(dfToFill) - 1, prefix='Progress:', suffix='Complete', length=50)
        counterRecoilLenght = 0
        counterNumberOfPasses = 0

        idPlayer = player[3]
        matchId = player[0]

        # take the match
        dfSubValue = dfEvent.loc[(dfEvent.matchId == matchId) & (dfEvent.playerId == idPlayer)]

        # for each event
        for passe in dfSubValue.values:
            # if we have a pass
            if (passe[1] == 'Pass'):
                # extract the direction of the pass
                x0 = passe[7][0]['x']
                y0 = passe[7][0]['y']
                x1 = passe[7][1]['x']
                y1 = passe[7][1]['y']

                # compute prodotto cartesiano
                absoluteX = abs(x0 - x1)
                absoluteY = abs(y0 - y1)
                summ = absoluteX ** 2 + absoluteY ** 2
                distance = math.sqrt(summ)
                # append
                counterRecoilLenght = counterRecoilLenght + distance
                counterNumberOfPasses = counterNumberOfPasses + 1
        # total lenght divided by the number of passes
        meanRecoil = round(weird_division(counterRecoilLenght, counterNumberOfPasses), 2)
        listMeanPassesRecoil.append(meanRecoil)

    print('Computing Team Pass Mean Lenght...')
    # Initial call to print 0% progress
    printProgressBar(0, len(dfToFill) - 1, prefix='Progress:', suffix='Complete', length=50)
    for index, player in dfToFill.iterrows():
        printProgressBar(index, len(dfToFill) - 1, prefix='Progress:', suffix='Complete', length=50)
        counterRecoilLenght = 0
        counterNumberOfPasses = 0

        matchId = player[0]
        teamId = player[2]

        # take the match
        dfSubValue = dfEvent.loc[(dfEvent.matchId == matchId) & (dfEvent.teamId == teamId)]

        # for each event
        for passe in dfSubValue.values:
            # if we have a pass
            if (passe[1] == 'Pass'):
                # extract the direction of the pass
                x0 = passe[7][0]['x']
                y0 = passe[7][0]['y']
                x1 = passe[7][1]['x']
                y1 = passe[7][1]['y']

                # compute prodotto cartesiano
                absoluteX = abs(x0 - x1)
                absoluteY = abs(y0 - y1)
                summ = absoluteX ** 2 + absoluteY ** 2
                distance = math.sqrt(summ)
                # append
                counterRecoilLenght = counterRecoilLenght + distance
                counterNumberOfPasses = counterNumberOfPasses + 1
        # total lenght divided by the number of passes
        meanRecoil = round(weird_division(counterRecoilLenght, counterNumberOfPasses), 2)
        listMeanPassesRecoilTeam.append(meanRecoil)

    print('Computing Team Pass Mean Lenght...')
    # Initial call to print 0% progress
    printProgressBar(0, len(dfToFill) - 1, prefix='Progress:', suffix='Complete', length=50)
    for index, player in dfToFill.iterrows():
        printProgressBar(index, len(dfToFill) - 1, prefix='Progress:', suffix='Complete', length=50)
        counterRecoilLenght = 0
        counterNumberOfPasses = 0

        matchId = player[0]

        # take the match
        dfSubValue = dfEvent.loc[(dfEvent.matchId == matchId)]

        # for each event
        for passe in dfSubValue.values:
            # if we have a pass
            if (passe[1] == 'Pass'):
                # extract the direction of the pass
                x0 = passe[7][0]['x']
                y0 = passe[7][0]['y']
                x1 = passe[7][1]['x']
                y1 = passe[7][1]['y']

                # compute prodotto cartesiano
                absoluteX = abs(x0 - x1)
                absoluteY = abs(y0 - y1)
                summ = absoluteX ** 2 + absoluteY ** 2
                distance = math.sqrt(summ)
                # append
                counterRecoilLenght = counterRecoilLenght + distance
                counterNumberOfPasses = counterNumberOfPasses + 1
        # total lenght divided by the number of passes
        meanRecoil = round(weird_division(counterRecoilLenght, counterNumberOfPasses), 2)
        listMeanPassesRecoilMatch.append(meanRecoil)

    dfToFill['mean_lenght_passes_individual'] = listMeanPassesRecoil
    dfToFill['mean_lenght_passes_team'] = listMeanPassesRecoilTeam
    dfToFill['mean_lenght_passes_match'] = listMeanPassesRecoilMatch

    return dfToFill


def centerOfGravity(df, dfEvent):
    '''
    This function compute the mean position of each player inside a match, morover count the mean position of the whole team
    and the the mean position of the whole match
    Params:
        df that contains all the information regarding player and game week and other, so a dataset
            that at this point in computation count more or less 133 columns
        df event the json dataset that store all the games event

    Return:
        df that contains 139 columns with the mean position feature added
    '''
    listMeanCenterGravityX = []
    listMeanCenterGravityY = []
    listMeanCenterGravityTeamX = []
    listMeanCenterGravityTeamY = []
    listMeanCenterGravityMatchX = []
    listMeanCenterGravityMatchY = []

    print('Computing Indivudal Mean Gravity...')
    # Initial call to print 0% progress
    printProgressBar(0, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)

    for index, player in df.iterrows():
        printProgressBar(index, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)

        counterMeanCenterGravityX = 0
        counterMeanCenterGravityY = 0
        counterNumberPosition = 0

        idPlayer = player[3]
        matchId = player[0]

        # take the match
        dfSubValue = dfEvent.loc[(dfEvent.matchId == matchId) & (dfEvent.playerId == idPlayer)]

        # for each event
        for passe in dfSubValue.values:
            # extract the position of the starting event
            x0 = passe[7][0]['x']
            y0 = passe[7][0]['y']
            counterMeanCenterGravityX = counterMeanCenterGravityX + x0
            counterMeanCenterGravityY = counterMeanCenterGravityY + y0
            counterNumberPosition = counterNumberPosition + 1

        # total positions divided by the number of events
        meanGravityX = round(weird_division(counterMeanCenterGravityX, counterNumberPosition), 2)
        meanGravityY = round(weird_division(counterMeanCenterGravityY, counterNumberPosition), 2)
        listMeanCenterGravityX.append(meanGravityX)
        listMeanCenterGravityY.append(meanGravityY)

    print('Computing Team Mean Gravity...')
    # Initial call to print 0% progress
    printProgressBar(0, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)

    for index, player in df.iterrows():
        printProgressBar(index, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)

        counterMeanCenterGravityX = 0
        counterMeanCenterGravityY = 0
        counterNumberPosition = 0

        matchId = player[0]
        teamId = player[2]

        # take the match
        dfSubValue = dfEvent.loc[(dfEvent.matchId == matchId) & (dfEvent.teamId == teamId)]

        # for each event
        for passe in dfSubValue.values:
            # extract the position of the starting event
            x0 = passe[7][0]['x']
            y0 = passe[7][0]['y']
            counterMeanCenterGravityX = counterMeanCenterGravityX + x0
            counterMeanCenterGravityY = counterMeanCenterGravityY + y0
            counterNumberPosition = counterNumberPosition + 1

        # total position divided by the number of event
        meanGravityX = round(weird_division(counterMeanCenterGravityX, counterNumberPosition), 2)
        meanGravityY = round(weird_division(counterMeanCenterGravityY, counterNumberPosition), 2)
        listMeanCenterGravityTeamX.append(meanGravityX)
        listMeanCenterGravityTeamY.append(meanGravityY)

    print('Computing Match Mean Gravity...')
    # Initial call to print 0% progress
    printProgressBar(0, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)

    for index, player in df.iterrows():
        printProgressBar(index, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)

        counterMeanCenterGravityX = 0
        counterMeanCenterGravityY = 0
        counterNumberPosition = 0

        matchId = player[0]

        # take the match
        dfSubValue = dfEvent.loc[(dfEvent.matchId == matchId)]

        # for each event
        for passe in dfSubValue.values:
            # extract the starting position of the event
            x0 = passe[7][0]['x']
            y0 = passe[7][0]['y']
            counterMeanCenterGravityX = counterMeanCenterGravityX + x0
            counterMeanCenterGravityY = counterMeanCenterGravityY + y0
            counterNumberPosition = counterNumberPosition + 1

        # total position divided by the number of event
        meanGravityX = round(weird_division(counterMeanCenterGravityX, counterNumberPosition), 2)
        meanGravityY = round(weird_division(counterMeanCenterGravityY, counterNumberPosition), 2)
        listMeanCenterGravityMatchX.append(meanGravityX)
        listMeanCenterGravityMatchY.append(meanGravityY)

    df['x_individual_mean_center_of_gravity'] = listMeanCenterGravityX
    df['y_individual_mean_center_of_gravity'] = listMeanCenterGravityY
    df['x_team_mean_center_of_gravity'] = listMeanCenterGravityTeamX
    df['y_team_mean_center_of_gravity'] = listMeanCenterGravityTeamY
    df['x_match_mean_center_of_gravity'] = listMeanCenterGravityMatchX
    df['y_match_mean_center_of_gravity'] = listMeanCenterGravityMatchY

    return df


# 52 #53 %54
def computeEjectionFeature(df):
    '''
    This function compute if the player is ejected during the game
    Params:
        df of the player that needed to be filled

    Return:
        df with the feature containing the ejection feature
    '''
    # exctract the array of red card
    listRed = df['red_card_features_extarction']
    # exctract the array of second yello card
    list2y = df['second_yellow_card_features_extarction']
    listEjection = []
    # Initial call to print 0% progress
    printProgressBar(0, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)
    # for each player
    for index, player in df.iterrows():
        printProgressBar(index, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)
        # do the sum
        listEjection.append(listRed[index] + list2y[index])

    df['ejection'] = listEjection

    return df


def computeCountryOfThePlayer(df, playersDataframe):
    '''
    This function compute the player country
    Params:
        df of the player that needed to be filled with the country
        playersDataFrame that contains the dataframe of the player json

    Return:
        df with the computation of the country feature
    '''
    listOfNationalities = []
    for playerGame in df.values:
        matched = False
        for player in playersDataframe.values:
            if (playerGame[3] == player[16]):
                listOfNationalities.append(player[0]['name'])
                matched = True
        if(matched == False):
            listOfNationalities.append('not_know')
    df['country'] = listOfNationalities
    return df


def computeBigMatchFeature(df, matches):
    '''
    This function compute if the match the player played is a big match and moreover compute if the game is a winning one
    for each combination match player

    Params:
        df of the player that needed to be filled
        matches dataset that contains the match json file from wyscout

    Return:
        df with the feature containing the match winning and big match feature
    '''

    listBigMatch = []
    listWinner = []

    print('Computing Contextal Variables (Big Match and Winning Feature)...')
    # Initial call to print 0% progress
    printProgressBar(0, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)
    for index, player in df.iterrows():
        printProgressBar(index, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)
        # for each match
        for el in matches.values:
            # equal match id
            if (player[0] == el[13]):

                ##BIG MATCH PART

                # flag that says if one of two are found
                firstInBig = False
                secondInBig = False
                # for each team id we check if it is in the set
                for team in el[10]:
                    toCompare = int(team)
                    if (toCompare in bigMatch):
                        if (firstInBig):
                            secondInBig = True
                        else:
                            firstInBig = True
                if (firstInBig and secondInBig):
                    listBigMatch.append(1)
                else:
                    listBigMatch.append(0)
                ######################################

                ## WINNER PART
                # if the team id player is equal to the winner one
                if (player[2] == el[12]):
                    # we append 1
                    listWinner.append(1)
                else:
                    # else we append 0
                    listWinner.append(0)

    df['big_match'] = listBigMatch
    df['winner'] = listWinner

    return df


import re


def computeLastContextualVariables(df, dfMatches, dfTeams):
    '''
    This function compute the remaining variables that needed to be computed, goal suffered, team goal and goal difference

    Params:
        df the dataset with the combination player matches
        dfMatches the dataset that contains the matches json
        dfTeams the dataset that contains the teams json (needed to connect the team id to the team name)

    Return:
        df the dataset with the contextual variable inserted.
    '''
    listHomeAway = []
    listGoalMade = []
    listGoalSuffered = []
    listDifference = []
    listClubName = []
    listAgainstClubName = []

    teamsDictionary = createAssociationNameIdTeam(dfTeams)

    print('Computing Contextal Variables (H/A and Goal Statistics)...')
    # Initial call to print 0% progress
    printProgressBar(0, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)
    # for each player match combination
    for index, playerMatch in df.iterrows():
        printProgressBar(index, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)
        # for eahc match
        for match in dfMatches.values:
            # equal match id
            if (playerMatch[0] == match[13]):
                # we take the label
                label = match[5]
                label = label.replace(',', '-')
                label = label.split('-')
                for tok in label:
                    tok = re.sub(r"[^a-zA-Z0-9]+", '', tok)
                homeTeam = label[0]
                homeTeam = homeTeam[:-1]
                awayTeam = label[1]
                awayTeam = awayTeam[1:]
                scoreHome = int(label[2])
                scoreAway = int(label[3])
                # home check
                if (teamsDictionary[playerMatch[2]] == homeTeam):
                    listHomeAway.append(1)
                    listGoalMade.append(scoreHome)
                    listGoalSuffered.append(scoreAway)
                    listDifference.append(scoreHome - scoreAway)
                    listClubName.append(homeTeam)
                    listAgainstClubName.append(awayTeam)
                # not home
                elif (teamsDictionary[playerMatch[2]] == awayTeam):
                    listHomeAway.append(0)
                    listGoalMade.append(scoreAway)
                    listGoalSuffered.append(scoreHome)
                    listDifference.append(scoreAway - scoreHome)
                    listClubName.append(awayTeam)
                    listAgainstClubName.append(homeTeam)
                else:
                    listHomeAway.append(-1)
                    listGoalMade.append(-1)
                    listGoalSuffered.append(-1)
                    listDifference.append(-1)

    df['contextual_h/a'] = listHomeAway
    df['contextual_team_goal'] = listGoalMade
    df['contextual_goal_suffered'] = listGoalSuffered
    df['contextual_goal_difference'] = listDifference
    df['contextual_club_name'] = listClubName
    df['contextual_against_club_name'] = listAgainstClubName

    return df


def createAssociationNameIdTeam(dfTeams):
    dictionaryIdName = {}
    for team in dfTeams.values:
        if (team[7] == 'club'):
            if (team[0]['name'] == 'Italy'):
                dictionaryIdName[team[8]] = team[5]
    return dictionaryIdName


from datetime import date


def computeAgeForPlayer(df, dfPlayer):
    '''
    This function take a whole dataset of combination player match and look into dfPLayer dataset in order
    to retrive the age for each player

    Params:
        df player, match combination dataset
        df player dataset of wyscout json

    Return:
        df player, match combination dataset with the age variable added.
    '''
    listAge = []

    # retrive today date
    today = date.today()

    d1 = today.strftime("%Y/%m/%d")

    d1 = d1.replace('/', '-')

    d1 = d1.split('-')

    dToday = int(d1[0])

    print('Computing Contextal Variables (Age)...')
    # Initial call to print 0% progress
    printProgressBar(0, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)

    for index, playerMatch in df.iterrows():
        printProgressBar(index, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)
        matched = False
        for player in dfPlayer.values:
            # same id
            if (playerMatch[3] == player[16]):
                birth = player[1]
                birth = birth.split('-')
                birth = int(birth[0])

                listAge.append(dToday - birth)
                matched = True
        if(matched == False):
            listAge.append(20)
    df['contextual_age'] = listAge

    return df


def computeExpectation(df, dfbetting):
    '''
    This function take the dataset of player match paring and then, using the betting dataset add the expectation odd on that team

    Params:
        df the player, match paring dataset
        dfbetting the dataset that contain all the features regarding the betting of the game, expectation from bookmaker
    '''
    listBetting = []
    print('Computing Contextal Variables (Expectation)...')
    # Initial call to print 0% progress
    printProgressBar(0, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)

    for index, playerMatch in df.iterrows():
        printProgressBar(index, len(df) - 1, prefix='Progress:', suffix='Complete', length=50)

        playerMatchHomeTeam = playerMatch[148]
        playerMatchAwayTeam = playerMatch[149]

        playerMatchHomeTeam = playerMatchHomeTeam.upper()
        playerMatchAwayTeam = playerMatchAwayTeam.upper()

        if (playerMatchHomeTeam == 'INTERNAZIONALE'):
            playerMatchHomeTeam = 'INTER'

        if (playerMatchHomeTeam == 'HELLAS VERONA'):
            playerMatchHomeTeam = 'VERONA'

        if (playerMatchAwayTeam == 'INTERNAZIONALE'):
            playerMatchAwayTeam = 'INTER'

        if (playerMatchAwayTeam == 'HELLAS VERONA'):
            playerMatchAwayTeam = 'VERONA'

        odds = 0

        for odd in dfbetting.values:
            # if the home team name and the away team name are the same, we append the home team winning odd
            if (playerMatchHomeTeam == odd[2].upper() and playerMatchAwayTeam == odd[3].upper()):
                odds = odd[22]
        if (odds != 0):
            listBetting.append(odds)
        else:
            listBetting.append(-1)

    df['contextual_expecatition'] = listBetting

    return df


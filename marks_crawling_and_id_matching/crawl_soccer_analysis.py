import os
from time import sleep
import requests
import glob
import re
import pandas as pd

#########################################################################################
#############                                       #####################################
############# CRAWLING PART FOR FANTACALCIO RATINGS #####################################
#############                                       #####################################
#########################################################################################
from unidecode import unidecode


def crawlMarks(days, season, t, paths):
    """
    Crawl marks from fantacalcio sites and store the csv obtained into a folder called marks.
    Parameters
    ----------
    days: the list of gameweek we want to crawl in the gameweek
    season: string representing the season we want to crawl
    t: fixed value

    Returns
    -------
    None
    """
    # Create marks if does not exist
    path = paths

    if not os.path.exists(path + 'marks'):
        os.makedirs(path + 'marks')

    # URL Fantacalcio
    # parameter g indicates the day, parameter s indicate the season, parameter t a particular token for this season
    for day in days:
        # access the url
        url = 'http://www.fantacalcio.it/Servizi/Excel.ashx?type=1&g=' + str(day) + '&t=' + t + '&s=' + season + ''
        # take the request
        r = requests.get(url, allow_redirects=True)
        # check the lenght of the day, just for name ordering purpouse
        if (len(str(day)) > 1):
            open(path + 'marks/' + season + '-' + str(day) + '.xlsx', 'wb').write(r.content)
            sleep(0.5)
        else:
            # add a 0 if the number has lenght equal to 1
            open(path + 'marks/' + season + '-0' + str(day) + '.xlsx', 'wb').write(r.content)
            sleep(0.5)


# Function that checks if a string contains numbers.
def hasNumbers(inputString):
    """
    Check if a string has number inside
    Parameters
    ----------
    String

    Returns
    -------
    Boolean
        Return true if the string has number in input.
    """
    return any(char.isdigit() for char in inputString)

def merge_fantacalcio_scraped_ratings(paths):
    '''
    Take all the files inside the fantacalcio marks folder and create a dataset
    The merging phase take all the xlsx file inside the marks folder and merge
    all the records inside a single dataframe extracting the relevant information for our task.
    :return:
    Dataset composed by gameweek, player, team, role and rating
    '''
    # path where the fantacalcio marks are located
    path = os.path.join(paths, "marks")

    playerList = []
    teamList = []
    daysList = []
    marksList = []
    positionList = []

    giornata = 1
    # For each file inside the path with xlsx format
    for filename in glob.glob(os.path.join(path, '*.xlsx')):
        print(filename)
        # create a dataframe
        df = pd.read_excel(filename)
        df = df.drop([0, 1, 2])
        team = ' '
        # access dataframe
        for player in df.values:
            # check if the line contain the team: uppercase, without the same inside the team and without the dot
            if (not (hasNumbers(str(player[0]))) and player[0].isupper and player[0] != team and (
                    "." not in player[0])):
                team = player[0]
            # now we go for players
            if (hasNumbers(str(player[0]))):
                # we don't care about coach
                if (player[1] != 'ALL'):
                    # populate all lists
                    playerList.append(player[2])
                    positionList.append(player[1])
                    marksList.append(player[3])
                    teamList.append(team)
                    daysList.append(giornata)
        # update the day for the new xlsx
        giornata = giornata + 1

    newmarksList = []
    # rewrite those marks that contains "*" and cast them into float.
    for el in marksList:
        if (type(el) is not str):
            newmarksList.append(el)
        if (type(el) is str):
            el = float(re.sub('[^A-Za-z0-9]+', '', el))
            newmarksList.append(el)
    dataset = pd.DataFrame({'team': teamList, 'match_day': daysList, 'player': playerList, 'position': positionList,
                            'fantacalcio_score': newmarksList})
    return dataset


#########################################################################################
#############                                       #####################################
#############   CRAWLING PART FOR OTHER RATINGS     #####################################
#############                                       #####################################
#########################################################################################

def nameRemap(arrayPlayer):
    """
    Compute some preprocessing for each player name in order to retrive more clear and easy to use names
    There are same names that are inconsistent (for some reason the last letter of the name insert is a V)
    Parameters
    ----------
    List : arrayPlayer
        List of player names.

    Returns
    -------
    List
        List of processed player names.
    """
    newPlayerList = []
    # for each player
    for el in arrayPlayer:
        # split the tokens
        tokens = el.split()
        newTokens = []
        # for each token remove the ' '
        for tok in tokens:
            tok = tok.replace(' ', '')
            newTokens.append(tok)
        # if the lenght of the list of tokens is higher than 2 we remove the last element if is a single v
        if (len(newTokens) > 2 and (newTokens[len(newTokens) - 1] == 'V' or newTokens[len(newTokens) - 1] == 'P')):
            newTokens.pop()
            newString = ''
            first = True
            for part in newTokens:
                if (first):
                    newString = part
                    first = False
                else:
                    newString = newString + ' ' + part
            newPlayerList.append(newString)
        else:
            newString = ''
            first = True
            for part in newTokens:
                if (first):
                    newString = part
                    first = False
                else:
                    newString = newString + ' ' + part
            newPlayerList.append(newString)
    return newPlayerList


def marksManager(listOfScore):
    """
    For each mark we make a round to the first digit after the pointer and cast the mark to float
    Replace , with . and cast phase
    Parameters
    ----------
    List : listOfScore
        List of player Marks.

    Returns
    -------
    List of Float
        List of processed player marks rounded to 1 digit after the pointer.
    """
    newListOfScore = []
    for score in listOfScore:
        score = round(float(score), 1)
        newListOfScore.append(score)
    return newListOfScore

def merge_all_the_others_downloaded_ratings(path):
    '''
    this script take the path of the folder where are stored marks of the others ratings
    and for the rating of the season create the dataset
    :param path:
        path where is stored the csv file that contains the pianeta fanta ratings.
    :return:
        Dataset composed by player name, team, role, gameweek and ratings
    '''
    # path where the fantacalcio marks are located
    path =path
    playerList = []
    teamList = []
    daysList = []
    marksGazzettaList = []
    marksCorriereList = []
    marksTuttoSportList = []
    positionList = []

    # For each file inside the path with xlsx format
    for filename in glob.glob(os.path.join(path, '*.csv')):
        print(filename)
        df = pd.read_csv(filename, decimal=",", encoding="ISO-8859-1")
        # access dataframe
        for player in df.values:
            # discard those element that have at least one sv
            if (player[4] != 0 and player[9] != 0 and player[14] != 0 and player[4] != 's.v.' and player[
                9] != 's.v.' and player[14] != 's.v.'):
                playerList.append(player[3].replace('.', ''))
                positionList.append(player[0])
                marksGazzettaList.append(player[4])
                marksCorriereList.append(player[9])
                marksTuttoSportList.append(player[14])
                teamList.append(player[2].upper())
                daysList.append(player[1])
    playerList = nameRemap(playerList)
    newMarksGazzettaList = marksManager(marksGazzettaList)
    newMarksCorriereList = marksManager(marksCorriereList)
    newMarksTSList = marksManager(marksTuttoSportList)
    dataset = pd.DataFrame({'team': teamList, 'match_day': daysList, 'player': playerList, 'position': positionList,
                            'gazzetta_score': newMarksGazzettaList, 'corriere_score': newMarksCorriereList,
                            'tuttosport_score': newMarksTSList})
    return dataset


#########################################################################################
#############                                       #####################################
#############   Reordering part WYSCOUT DATA        #####################################
#############                                       #####################################
#########################################################################################
def unidecode_special_character(listato):
    '''
    Unidecode special character
        Applied for:
        * short names
        * names
        * last names
    :param listato:
        list of string to unidecode
    :return:
        unidecoded list
    '''
    new = []
    for el in listato:
        newEl = unidecode(el)
        new.append(newEl)
    return new

def generate_csv_for_string_matching_from_wyscout_jsons(dfteam, dfplayer):
    '''
    Generate a csv composed by team id, payer id, name, shaort name, role and last name in order to
    proceed to the last phase of crawling (string matching)
    :param dfteam:
        teams.json from wyscout
    :param dfplayer:
        players.json from wyscout
    :return:
    '''
    #Retrive Nationality for each team in order to obtain the id of the team
    listTeam = []
    listID = []
    checkItalianNationality = set()
    for team in dfteam.values:
        if (team[0]['name'] == 'Italy'):
            listTeam.append(team[5].upper())
            listID.append(team[8])
            checkItalianNationality.add(team[8])

    # REMAP HELLAS VERONA TO VERONA AND INTERNAZIONALE TO INTER
    newTeam = []
    for el in listTeam:
        if (el == 'HELLAS VERONA'):
            newTeam.append('VERONA')

        elif (el == 'INTERNAZIONALE'):
            newTeam.append('INTER')

        else:
            newTeam.append(el)

    dfItalianTeam = pd.DataFrame({'id_team': listID, 'team_name': newTeam})

    #Look for player that are inside the italian team.
    #For each player we check if the team is inside an italian team,
    # in that case we take all his information

    listIDPlayer = []
    listFirstName = []
    listIDTeamPlayer = []
    listLastName = []
    listShort = []
    listRole = []

    for player in dfplayer.values:
        if (player[3] in checkItalianNationality):
            listIDTeamPlayer.append(player[3])
            listFirstName.append(player[4].upper())
            listLastName.append(player[9].upper())
            listShort.append(player[13].upper().replace('.', ''))
            listIDPlayer.append(player[16])
            listRole.append(player[12]['code3'])

    #Change names in order to invert name and surname
    # Create a name based on surname and first name first letter
    newShort = []
    contator = 0
    for cognome in listLastName:
        newNome = cognome + " " + listFirstName[contator][0]
        contator = contator + 1
        newShort.append(newNome)
    '''=========================================================================='''
    # if the short name is smaller than the precreated name we prefer always the shortest
    # if the short is equal we prefer the new one
    secondNewShort = []
    index = 0
    for short in listShort:
        shorty = short.split()
        newshorty = newShort[index].split()
        if (len(shorty) < len(newshorty)):
            secondNewShort.append(short)
        if (len(shorty) == len(newshorty)):
            secondNewShort.append(newShort[index])
        if (len(shorty) > len(newshorty)):
            secondNewShort.append(short)
        index = index + 1
    '''=========================================================================='''

    # if the first part of the name is a single letter we want it at the end
    fourtNewShort = []
    for el in secondNewShort:
        newEl = el.split()
        # if the first word of the name is a single letter
        if (len(newEl[0]) == 1):
            newName = newEl[1] + ' ' + newEl[0]
            fourtNewShort.append(newName)
        else:
            fourtNewShort.append(el)

    newRole = []
    for role in listRole:
        if (role == 'GKP'):
            newRole.append('P')
        if (role == 'DEF'):
            newRole.append('D')
        if (role == 'MID'):
            newRole.append('C')
        if (role == 'FWD'):
            newRole.append('A')
    listRole = newRole

    #UNIDECODE SPECIAL CHAR APPLYIED FOR SHORT NAMES, NAMES AND LAST NAMES
    fiveNewShort = unidecode_special_character(fourtNewShort)
    newListFirstName = unidecode_special_character(listFirstName)
    newListLastName = unidecode_special_character(listLastName)
    dfPlayerID = pd.DataFrame(
        {'id_team': listIDTeamPlayer, 'id_player': listIDPlayer, 'short': fiveNewShort, 'first': newListFirstName,
         'last': newListLastName, 'role': listRole})
    df_merge_col = pd.merge(dfPlayerID, dfItalianTeam, on='id_team')
    return df_merge_col
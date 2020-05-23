# import dependencies
import json
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from config import db_password
import time

# This function takes in 3 arguments from loaded datasets extracted from wikipedia and kaggle. 
# The funciton extracts, cleans, and transforms relevant data and then uploads it to a SQL database.
# This function is called later in the code.
def transform_movies(wiki_movies_raw, kaggle_metadata, ratings):
    # Transform Wikipedia data
    # remove items that are not movies from wiki movies list
    # this line assumes these three arguments remain valid ones for screening out NON-movies from wikipedia. 
    # If the wikipedia data format changes, different criteria may need to be used to screen out non-movies.
    wiki_movies = [movie for movie in wiki_movies_raw
                if ('Director' in movie or 'Directed by' in movie)
                    and 'imdb_link' in movie
                    and 'No. of episodes' not in movie]

    # define funciton to clean individual movies in wikipedia file
    def clean_movie(movie):
        movie = dict(movie) #create a non-destructive copy
        alt_titles = {}
        # combine alternate titles into 1 list
        # This loop assumes that the listed keys for alternatives titles are the only ones in the wikipedia dataset
        # because this is what was in the original dataset. If new keys (such as a new language) get added, they will
        # need to be added to the list for the for loop in order move everything into the new alt_title category.
        for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                    'Hangul','Hebrew','Hepburn','Japanese','Literally',
                    'Mandarin','McCune–Reischauer','Original title','Polish',
                    'Revised Romanization','Romanized','Russian',
                    'Simplified','Traditional','Yiddish']:
            if key in movie:
                alt_titles[key] = movie[key]
                movie.pop(key)
        if len(alt_titles) > 0:
            movie["alt_titles"] = alt_titles
        
        # define function to merge column names
        def change_column_name(old_name, new_name):
            if old_name in movie:
                movie[new_name] = movie.pop(old_name)

        # call column name change function for a series of redundant column names 
        # This series of column name changes assumes that these are the relevant columns that are duplicated
        # and need to be made consistent. If new wikipedia data includes new or different columns they may need
        # to be added here.   
        change_column_name('Adaptation by', 'Writer(s)')
        change_column_name('Country of origin', 'Country')
        change_column_name('Directed by', 'Director')
        change_column_name('Distributed by', 'Distributor')
        change_column_name('Edited by', 'Editor(s)')
        change_column_name('Length', 'Running time')
        change_column_name('Original release', 'Release date')
        change_column_name('Music by', 'Composer(s)')
        change_column_name('Produced by', 'Producer(s)')
        change_column_name('Producer', 'Producer(s)')
        change_column_name('Productioncompanies ', 'Production company(s)')
        change_column_name('Productioncompany ', 'Production company(s)')
        change_column_name('Released', 'Release Date')
        change_column_name('Release Date', 'Release date')
        change_column_name('Screen story by', 'Writer(s)')
        change_column_name('Screenplay by', 'Writer(s)')
        change_column_name('Story by', 'Writer(s)')
        change_column_name('Theme music composer', 'Composer(s)')
        change_column_name('Written by', 'Writer(s)')
        
        return movie

    # call clean movie function for each movie in list and save to new list    
    clean_movies = [clean_movie(movie) for movie in wiki_movies]

    # convert clean wiki movies to dataframe
    wiki_movies_df = pd.DataFrame(clean_movies)

    # Use regular expression to extract IMDB ID from the IMDB link and drop any duplicates
    wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
    wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)

    # drop any columns with data less than 10% of real entries (< 90% of data is null)
    # this analysis assumes that 10% is an appropriate cutoff for amount of data to exist. 
    # This can be adjusted in the future by changing 0.9 to a different cut off.
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

    # Box Office Transformation
    # extract a list of Box office data and remove NaNs
    box_office = wiki_movies_df['Box office'].dropna() 
    # remove any data stored as lists
    box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)
    # clean values given as a range
    # This will replace with the UPPER (2nd) end of the range. The analysis assumes the range 
    # is relatively small and thus this will not have a big effect but this may need to be revisited
    # if ranges in new datasets are large (ie $100-500 million) because then taking the upper end of the
    # range could skew the data.
    box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    # define 2 generic forms for money data 
    # this analysis assumes these two regular expression forms will capture the vast majority 
    # of the data as was the case during exploratory analysis. If other forms become
    # prevalent, they may need to be added here.
    form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
    form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'

    #define function to turn extracted strings into numbers
    # this function assumes million and billion are the relevant parsings that need to be done but in the future
    # others may be needed (ie "thousand" or "hundred thousand")
    def parse_dollars(s):
        # if s is not a string, return NaN
        if type(s) != str:
            return np.nan

        # if input is of the form $###.# million
        if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):
            # remove dollar sign and " million"
            s = re.sub('\$|\s|[a-zA-Z]','', s)
            # convert to float and multiply by a million
            value = float(s) * 10**6
            return value

        # if input is of the form $###.# billion
        elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):
            # remove dollar sign and " billion"
            s = re.sub('\$|\s|[a-zA-Z]','', s)
            # convert to float and multiply by a billion
            value = float(s) * 10**9
            return value

        # if input is of the form $###,###,###
        elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):
            # remove dollar sign and commas
            s = re.sub('\$|,','', s)
            # convert to float
            value = float(s)
            return value

        # otherwise, return NaN
        else:
            return np.nan

    #Call Parse dollars function on Box_office and add to dataframe
    wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    # drop "dirty" box office data
    wiki_movies_df.drop('Box office', axis=1, inplace=True)

    # Clean Budget Data
    # create a list of budgets
    budget = wiki_movies_df['Budget'].dropna()
    # convert lists to strings
    budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)
    # convert ranges to a single value
    # as above, this assumes the range is relatively small and takes the UPPER end of the range so 
    # this may need to be changed if ranges in the data are large and taking the upper introduces skew to data.
    budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    #Call Parse dollars function on budget and add to dataframe
    wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    # drop "dirty" budget data
    wiki_movies_df.drop('Budget', axis=1, inplace=True)

    # clean release date data
    # extract non-null release date values and convert lists to strings
    release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

    # define generic forms for date data
    # this analysis assumes these regular expressions will capture the vast majority of the dates as they did in the initial
    # data set but the RegEx's may need to be adjusted if new date forms are introduced.
    date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
    date_form_two = r'\d{4}.[01]\d.[123]\d'
    date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    date_form_four = r'\d{4}'

    # Extract values with match regular expressions, convert to DateTime, add to dataframe
    wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)
    # drop "dirty" relase date data
    wiki_movies_df.drop('Release date', axis=1, inplace=True)

    # clean runtime data
    # create list of non-null runtime and convert lists to strings
    running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    # extract values with regular expression
    running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')
    # convert values to numeric
    running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
    # convert to pure minuts and add to dataframe
    wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)
    # drop "dirty" running time data
    wiki_movies_df.drop('Running time', axis=1, inplace=True)


    # Transform Kaggle data
    # drop adult column and any adult movies
    kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')
    # convert video column to Boolean
    kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'
    # convert data to numeric
    kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
    kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
    kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')
    # convert data to DateTime
    kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])

    # Transform ratings data to datetime
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')


    # MERGE dataframes
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])

    # this analysis makes assumptions about how to handle duplicated data (see table below) that came frmo our exploratory data analysis
    # if data integrity changes, these may need to be revisited and handled in different ways.

    # DataMerge Plan (from original exploratory analysis):
    # Wiki                     Movielens                Resolution
    #--------------------------------------------------------------------------
    # title_wiki               title_kaggle            Drop Wikipedia
    # running_time             runtime                Keep Kaggle; fill in 0s
    # budget_wiki              budget_kaggle          Keep Kaggle; fill in 0s
    # box_office               revenue                Keep Kaggle; fill in 0s
    # release_date_wiki        release_date_kaggle     Drop Wikipedia
    # Language                 original_language       Drop Wikipedia
    # Production company(s)    production_companies    Drop Wikipedia

    # Drop wikipedia columns
    movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)

    # define function to keep kaggle data but fill in with Wikipedia data if available.
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
            , axis=1)
        df.drop(columns=wiki_column, inplace=True)

    # Call kaggle fill function
    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')    

    # drop any columns with only 1 value
    for col in movies_df.columns:
        lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
        value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
        num_values = len(value_counts)
        if num_values == 1:
            movies_df[col].value_counts(dropna=False)  

    # reorder columns
    # this process assumes the listed columns are the ones interested in. If others, should be added to the code.
    movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                        'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                        'genres','original_language','overview','spoken_languages','Country',
                        'production_companies','production_countries','Distributor',
                        'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                        ]]   

    # rename columns
    # this renaming assumes the listed columns are the only ones in the dataset. If others get added (above) they'll 
    # need to be added to here to rename as well.
    movies_df.rename({'id':'kaggle_id',
                    'title_kaggle':'title',
                    'url':'wikipedia_url',
                    'budget_kaggle':'budget',
                    'release_date_kaggle':'release_date',
                    'Country':'country',
                    'Distributor':'distributor',
                    'Producer(s)':'producers',
                    'Director':'director',
                    'Starring':'starring',
                    'Cinematography':'cinematography',
                    'Editor(s)':'editors',
                    'Writer(s)':'writers',
                    'Composer(s)':'composers',
                    'Based on':'based_on'
                    }, axis='columns', inplace=True)  

    # get rating counts for each movie ID
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \
                    .rename({'userId':'count'}, axis=1) \
                    .pivot(index='movieId',columns='rating', values='count')

    # rename columns
    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]

    #left merge with movies_df
    movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')
    # fill in missing values
    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)


    #connection infor for postgres
    db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"

    engine = create_engine(db_string)

    # add movies DF to sql. In order to handle errors, this is approached with a try except block.
    # if table already exists, will return a value error so the code will move to the except which 
    # involves "replacing" the table to overwrite old data.
    try:
        movies_df.to_sql(name='movies', con=engine)
    except ValueError:
        movies_df.to_sql(name='movies', con=engine, if_exists='replace')


    # Add ratings data to SQL.
    # need to figure out how to handle adding data. can't replace like above.
    rows_imported = 0
    # get the start_time from time.time()
    start_time = time.time()
    for data in pd.read_csv(f'{file_dir}/the-movies-dataset/ratings.csv', chunksize=1000000):
        print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')

        # on first pass through, try adding to sql, if ValueError due to existing table, replace table.
        if rows_imported == 0:
            #In order to handle errors, this is approached with a try except block.
            # if table already exists, will return a value error so the code will move to the except which 
            # involves "replacing" the table to overwrite old data.
            try:
                #data.to_sql(name='ratings', con=engine)
            except ValueError:
                #data.to_sql(name='ratings', con=engine, if_exists='replace')
                
        # after first iteration, append additional results onto table so we do not overwite data from previous blocks
        else:
            data.to_sql(name='ratings', con=engine, if_exists='append')
        rows_imported += len(data)

        # add elapsed time to final print out
        print(f'Done. {time.time() - start_time} total seconds elapsed')


# Load files program. this assumes the file_dir is set to the working directory and the wikipedia json file is in this same folder
# while the two csv's from kaggle are in a subfolder "the-movies-dataset." These paths should be edited if this changes.
# set file directory
try:
    file_dir = "/Users/Jack_Walker/Vanderbilt/Data-Analytics-Bootcamp/Week8-ETL/movie_hackathon/"
    # Extract wikipedia, kaggle, and movie rating files
    with open(f'{file_dir}wikipedia.movies.json', mode='r') as file:
        wiki_movies_raw = json.load(file)

    kaggle_metadata = pd.read_csv(f'{file_dir}the-movies-dataset/movies_metadata.csv')
    ratings = pd.read_csv(f'{file_dir}the-movies-dataset/ratings.csv')

    # call large functin to transform and load data
    transform_movies(wiki_movies_raw, kaggle_metadata, ratings)

except FileNotFoundError:
    print("Please update files or file paths in order to load data and run program.")

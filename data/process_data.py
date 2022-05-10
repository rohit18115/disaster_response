import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads messages and categories data from their respective filepath
    and merge them in a dataframe.

    Parameters:
        message_filepath: the file path for message data
        categories_filepath: the file path for categories

    Returns:
        Merged dataframe

    """
    messages = pd.read_csv(messages_filepath, ",")
    categories = pd.read_csv(categories_filepath, ",")
    df = pd.merge(messages, categories, on="id", how="outer")
    return df


def clean_data(df):
    """Cleans the merged dataframe containing the messages and categories.

    Parameters:
        df: The merged dataframe that needs to be cleaned

    Returns:
        Cleaned dataframe.

    """
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0, :]
    category_colnames = [i[:-2] for i in row]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [int(i[-1]) for i in categories[column]]
    # drop the original categories column from `df`
    df = df.drop(["categories"], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df, categories


def save_data(df, categories, database_filepath, categories_filepath):
    """Saves the cleaned dataframe to the specified path.

    Parameters:
        df: The clean dataframe that needs to be saved.
        database_filename: the filename of the clean dataframe df.

    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df.to_sql("disaster_texts", engine, index=False, if_exists="replace")
    cat_engine = create_engine("sqlite:///categories_database.db")
    categories.to_sql("categories", cat_engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df, categories = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, categories, database_filepath, categories_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()

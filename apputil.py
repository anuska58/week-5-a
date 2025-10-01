import plotly.express as px
import pandas as pd


def load_data(url=None):
    """Load dataset from a given URL of Titanic dataset."""
    if url is None:
        url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
    df = pd.read_csv(url)
    df = df.rename(columns=str.lower)
    return df


def survival_demographics(df=None):
    """
    Analyze survival patterns based on class, sex, and age group.
    """   
    if df is None:
        df = load_data()
    bins=[0,12,19,59,float('inf')]
    labels=['Child','Teen','Adult','Senior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

    grouped = df.groupby(['pclass', 'sex', 'age_group'])

    result=grouped['survived'].agg(
        n_passengers='count',
        n_survivors='sum',
        survival_rate='mean'
        ).reset_index()
    
    result=result.sort_values(by=['pclass','sex','age_group']).reset_index(drop=True)
    return result

def family_groups(df=None):
    """
    Analyze survival patterns based on family size.
    """
    if df is None:
        df = load_data()
    df['family_size'] = df['sibsp'] + df['parch'] + 1  # Including the passenger themselves
    grouped = df.groupby(['family_size', 'pclass'])["fare"]
    result = grouped.agg(
        n_passengers='count',
        avg_fare='mean',
        min_fare='min',
        max_fare='max'
        ).reset_index()
    result = result.sort_values(by=['pclass', 'family_size']).reset_index(drop=True)
    return result

def last_names(df=None):
    """
    Extract last names and count their occurences
    """
    if df is None:
        df = load_data()
    df['LastName'] = df['name'].apply(lambda x: x.split(',')[0].strip())
    last_name_counts = df['LastName'].value_counts()
    return last_name_counts

def determine_age_division(df=None):
    """
    Add a column 'older_passenger' to indicate if a passenger is older
    than the median age for their passenger class.
    """
    if df is None:
        df = load_data()
    median_by_class = df.groupby("pclass")["age"].transform("median")
    df["older_passenger"] = df["age"] > median_by_class
    df["older_passenger"] = df["older_passenger"].fillna(False) 
    return df

def visualize_demographic(df=None):
    """
        Visualize survival patterns based on demographics."""
    if df is None:
        df = load_data()
    demo_df = survival_demographics(df)
    fig = px.bar(
        demo_df,
        x='age_group',
        y='survival_rate',
        color="sex",
        barmode='group',
        facet_col='pclass',
        title='Survival Rate by Age Group, sex, and Passenger Class'
    )
    #fig.update_layout(yaxis_title='Survival Rate', xaxis_title='Age Group')
    return fig

def visualize_families(df=None):
    """
    Visualize survival patterns based on family size and wealth.
    """
    if df is None:
        df = load_data()
    table_df = family_groups(df)
    fig=px.line(
        table_df,
        x='family_size',
        y='avg_fare',
        color='pclass',
        markers=True,
        title='Average Fare by Family Size and Class'
    )
    return fig

def visualize_family_size(df=None):
    """
    Visualize the distribution of family sizes.
    """
    if df is None:
        df = load_data()
    table=family_groups(df)
    fig=px.bar(
       table,
       x='family_size',
       y='n_passengers',
       color='pclass',
       barmode='group',
       title='Number of Passengers by Family Size and Class'
   )
    return fig
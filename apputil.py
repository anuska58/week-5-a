import plotly.express as px
import pandas as pd


def load_data(url=None):
    """Load dataset from a given URL of Titanic dataset."""
    if url is None:
        url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
    df = pd.read_csv(url)
    return df


def survival_demographics():
    """
    Analyze survival patterns based on class, sex, and age group.
    """   
    df = load_data()
    bins=[0,12,19,59,float('inf')]
    labels=['Child','Teen','Adult','Senior']
    df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

    grouped = df.groupby(['Pclass', 'Sex', 'age_group'])

    result=grouped['Survived'].agg(
        n_passengers='count',
        n_survivors='sum',
        survival_rate='mean'
        ).reset_index()
    
    result=result.sort_values(by=['Pclass','Sex','age_group']).reset_index(drop=True)
    return result

def family_groups():
    """
    Analyze survival patterns based on family size.
    """
    df = load_data()
    df['family_size'] = df['SibSp'] + df['Parch'] + 1  # Including the passenger themselves
    grouped = df.groupby(['family_size', 'Pclass'])["Fare"]
    result = grouped.agg(
        n_passengers='count',
        avg_fare='mean',
        min_fare='min',
        max_fare='max'
        ).reset_index()
    result = result.sort_values(by=['Pclass', 'family_size']).reset_index(drop=True)
    return result

def last_names():
    """
    Extract last names and count their occurences
    """
    df = load_data()
    df['LastName'] = df['Name'].apply(lambda x: x.split(',')[0].strip())
    last_name_counts = df['LastName'].value_counts()
    return last_name_counts

def determine_age_division():
    """
    Add a column 'older_passenger' to indicate if a passenger is older
    than the median age for their passenger class.
    """
    df = load_data()
    median_by_class = df.groupby("Pclass")["Age"].transform("median")
    df["older_passenger"] = df["Age"] > median_by_class
    return df

def visualize_demographic():
    """
        Visualize survival patterns based on demographics."""
    df = load_data()
    demo_df = survival_demographics(df)
    fig = px.bar(
        demo_df,
        x='age_group',
        y='survival_rate',
        color="Sex",
        barmode='group',
        facet_col='Pclass',
        title='Survival Rate by Age Group, Sex, and Passenger Class'
    )
    #fig.update_layout(yaxis_title='Survival Rate', xaxis_title='Age Group')
    return fig

def visualize_families():
    """
    Visualize survival patterns based on family size and wealth.
    """
    df = load_data()
    table_df = family_groups(df)
    fig=px.line(
        table_df,
        x='family_size',
        y='avg_fare',
        color='Pclass',
        markers=True,
        title='Average Fare by Family Size and Class'
    )
    return fig

def visualize_family_size():
    """
    Visualize the distribution of family sizes.
    """
    df = load_data()
    table=family_groups(df)
    fig=px.bar(
       table,
       x='family_size',
       y='n_passengers',
       color='Pclass',
       barmode='group',
       title='Number of Passengers by Family Size and Class'
   )
    return fig
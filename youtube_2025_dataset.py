
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
    
# =================== Load Dataset ===================
file_path = "youtube_2025_dataset.csv"
df = pd.read_csv(file_path)

try:
    with open(file_path, mode ='r', encoding = 'utf-8') as file:
        csv_read = csv.reader(file)
        headers = next(csv_read)
        members_index = headers.index("Members Count")
        print("Headers:",headers)

        row_count = 0
        Members_count=0

        print("\nPreview of first 5 rows:")

        for row in csv_read:
            if row_count < 5:
                print(row)
            row_count += 1

            try:
                members = int(row[members_index].replace(',',' ').strip())
                if members > 0 :
                    Members_count += 1
            except:
                continue      
except FileNotFoundError:
    print("File not found.please check the path.")
except Exception as e:
    print("An error occured while reading the file:",str(e))
else:
    print(f"\n Total number of data rows (excluding header):{row_count}")

# =================== Rename Columns ===================

df.columns = df.columns.str.strip()  
df.rename(columns={
    "Youtuber Name": "Youtuber",
    "Total Videos": "Total_Videos",
    "Avg Video Length (min)": "Avg_Video_Length_Min",
    "Total Subscribers": "Subscribers",
    "Members Count": "Members_Count",
    "AI Generated Content (%)": "AI_Generated_Percent",
    "Neural Interface Compatible": "Neural_Interface",
    "Metaverse Integration Level": "Metaverse_Level",
    "Quantum Computing Topics": "Quantum_Topics",
    "Holographic Content Rating": "Holographic_Rating",
    "Engagement Score": "Engagement",
    "Content Value Index": "Content_Value",
    "Channel Name": "Channel_Name"
}, inplace=True)



# =================== Type Conversions ===================
metaverse_map = {'none': 0, 'basic': 1, 'advanced': 2, 'full': 3}

df['Metaverse_Level'] = df['Metaverse_Level'].astype(str).str.strip().str.lower()
df['Metaverse_Level'] = df['Metaverse_Level'].map(metaverse_map).fillna(0).astype(int)

df['Neural_Interface'] = df['Neural_Interface'].astype(str).str.strip().str.lower()
   # Create a mapping for the Holographic Content Rating
rating_map = {
    '1D': 1,
    '2D': 2,
    '3D': 3,
    '4D': 4
}

# Apply the mapping to the 'Holographic_Rating' column
df['Holographic_Rating_Numeric'] = df['Holographic_Rating'].map(rating_map)

# Convert numeric columns
num_cols = [
    "Total_Videos", "Subscribers", "Members_Count", "AI_Generated_Percent",
    "Quantum_Topics", "Avg_Video_Length_Min", "Holographic_Rating",
    "Engagement", "Content_Value"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# =================== Remove Invalid Rows ===================
df = df[(df["Subscribers"] > 0) & (df["Total_Videos"] > 0)]

# =================== MySQL Connection and Insert ===================
connect = mysql.connector.connect(
    host='127.0.0.1',
    port=3307,
    user='root',
    password='1234',
    database='Youtube_2025'
)

cursor = connect.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS youtube_stats (
    channel_name VARCHAR(255),
    youtuber VARCHAR(255),
    total_videos INT,
    best_video TEXT,
    avg_video_length_min FLOAT,
    subscribers BIGINT,
    members_count INT,
    ai_generated_percent INT,
    neural_interface TINYINT(1),
    metaverse_level INT,
    quantum_topics INT,
    holographic_rating FLOAT,
    engagement FLOAT,
    content_value FLOAT
);
''')
connect.commit()

insert_query = '''
INSERT INTO youtube_stats (channel_name, youtuber, total_videos, best_video,
avg_video_length_min, subscribers, members_count,
ai_generated_percent, neural_interface, metaverse_level,
quantum_topics, holographic_rating, engagement, content_value)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
'''
inserted_count=0
for _, row in df.iterrows():
    try:
        values = (
            row.get('Channel_Name'),
            row.get('Youtuber'),
            int(row.get('Total_Videos')),
            row.get('Best Video', None),
            float(row.get('Avg_Video_Length_Min')),
            int(row.get('Subscribers')),
            int(row.get('Members_Count')),
            int(row.get('AI_Generated_Percent')),
            int(row.get('Neural_Interface')),
            int(row.get('Metaverse_Level')),
            int(row.get('Quantum_Topics')),
            float(row.get('Holographic_Rating')),
            float(row.get('Engagement')),
            float(row.get('Content_Value'))
        )
        cursor.execute(insert_query, values)
        inserted_count += 1
    except Exception as e:
        print("Error inserting row:", e)
        print(row.to_dict())

connect.commit()
print(f"\n Successfully inserted {inserted_count} rows into youtube_stats.")

# =================== Analytics ===================
print("\n===== Summary Statistics =====")
print(df.describe(include='all'))

print("\n===== Top 10 Trending Videos by Engagement =====")
top_trending_videos = df.sort_values(by='Engagement', ascending=False).head(10)
print(top_trending_videos[['Youtuber', 'Best Video', 'Engagement']])

print("\n===== Category-wise Comparisons =====")
category_comparison = df.groupby('Channel_Name').agg(
    Total_Videos=('Total_Videos', 'sum'),
    Engagement_mean=('Engagement', 'mean'),
    Engagement_max=('Engagement', 'max')
).reset_index()
print(category_comparison)

print("\n===== Top 10 Youtubers by Total Videos =====")
top10_youtubers = df.groupby('Youtuber').agg({
    'Total_Videos': 'sum',
    'Engagement': 'max',
}).reset_index()

top10_youtubers['Best Video'] = top10_youtubers['Youtuber'].apply(
    lambda x: df[df['Youtuber'] == x].sort_values(by='Engagement', ascending=False).iloc[0]['Best Video']
)

top10_youtubers = top10_youtubers.sort_values(by='Total_Videos', ascending=False)
print(top10_youtubers[['Youtuber', 'Total_Videos', 'Best Video', 'Engagement']])

print("\n===== Categories With Most Total Videos =====")
category_video_count = df.groupby('Channel_Name')['Total_Videos'].sum().sort_values(ascending=False)
print(category_video_count)

# =================== Visualizations ===================
sns.set(style='whitegrid')

# Engagement vs Subscribers
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Subscribers', y='Engagement', hue='Metaverse_Level', palette='cool')
plt.title('Engagement vs Subscribers')
plt.xlabel('Subscribers')
plt.ylabel('Engagement Score')
plt.legend(title='Metaverse Level')
plt.grid(True)
plt.show()

# Top 10 Youtubers by Subscribers
top_youtubers = df.groupby('Youtuber')['Subscribers'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
top_youtubers.plot(kind='barh', color='orange')
plt.title('Top 10 Youtubers by Subscribers')
plt.xlabel('Subscribers')
plt.ylabel('Youtuber')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

# Neural Interface Compatibility Pie Chart
values = df['Neural_Interface'].value_counts()
labels = values.index.map({1: 'Compatible', 0: 'Not Compatible'})
plt.figure(figsize=(6,6))
plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
plt.title('Neural Interface Compatibility')
plt.show()

#HeatMap:Correlation Matrix

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only = True), annot = True , cmap= 'coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
           
# Create the stocket dataset


def create_tweet_dataframes(): 
    for ticker in sp500[0:]:
        print(ticker)
        if os.path.exists('/work/nlp/b.irving/stock/tweets_dataframes/' + ticker + '_clean.csv'):
            print('Clean tweet file already exists')
            continue
        # Define the directory where the JSON files are stored
        directory = '/work/nlp/b.irving/stock/newTweets/' + ticker

        # Initialize an empty list to store the data
        data = []

        # Loop through each file in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                date = filename.split('.')[0]  # Extract the date from the filename
                combined_text = ""
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:

                    for line in file:
                        try:
                            entry = json.loads(line.strip())
                            text = entry['text'].replace('\n', ' ')  # Remove '\n' characters
                            try:
                                if detect(text) == 'en':
                                    combined_text += text + " [SEP] "  # Combine English text entries with [SEP] token
                            except LangDetectException:
                                continue  # Skip lines that cause language detection errors
                        except json.decoder.JSONDecodeError:
                            continue  # Skip lines that cause JSONDecodeError
                if combined_text.strip():  # Check if combined_text is not empty
                    data.append({'date': date, 'text': combined_text.strip()})

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data)
        # Display the DataFrame
        print(df.head())

        # Optionally, save the DataFrame to a CSV file
        df.to_csv('/work/nlp/b.irving/stock/tweets_dataframes/' + ticker + '_clean.csv', index=False)
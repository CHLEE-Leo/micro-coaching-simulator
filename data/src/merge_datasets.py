# %%
import pandas as pd
meal_to_goal_NLG = pd.read_csv('/home/messy92/Leo/NAS_folder/Chatbots/micro_coach_chatbot/meal-goal-alignment/data/src/1. meal_to_goal (maybe-raw).csv', index_col=0)
cleaned_enriched_meal_to_goal = pd.read_csv('/home/messy92/Leo/NAS_folder/Chatbots/micro_coach_chatbot/meal-goal-alignment/data/src/cleaned_enriched_meal_to_goal_paper.csv', index_col=0)

merged_df = pd.merge(cleaned_enriched_meal_to_goal, meal_to_goal_NLG, how='left', on=['goal_id', 'user_id', 'meal_id'])

left_columns = ['goal_id', 'user_id', 'meal_id'] + [i for i  in list(merged_df.columns) if not '_y' in i]
left_columns_with_natural_language_text = left_columns + ['title_y'] + ['ingredients_y']
merged_df = merged_df[left_columns_with_natural_language_text]

left_columns_no_suffix = [i.replace('_x', '') if '_x' in i else i for i in list(merged_df.columns)]

merged_df.columns = left_columns_no_suffix


merged_df.to_csv('/home/messy92/Leo/NAS_folder/Chatbots/micro_coach_chatbot/meal-goal-alignment/data/src/5. cleaned_enriched_meal_to_goal_paper_with_natural_language_text.csv')
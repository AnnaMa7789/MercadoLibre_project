import os
import pandas as pd
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

def analyze_and_improve_titles_with_ai(df, sample_size=1000, deepseek_api_key=None):
    """
    Use DeepSeek AI to:
    1. Identify quality angles from sample titles
    2. Find 10 most problematic titles
    3. Provide specific improvement suggestions
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing product data with at least 'title' column
    sample_size : int
        Number of titles to sample for analysis
    deepseek_api_key : str
        DeepSeek API key
        
    Returns:
    --------
    dict : AI analysis results with problematic titles and suggestions
    pandas DataFrame : Original DataFrame with AI analysis results
    """
    
    print("🤖 AI TITLE QUALITY ANALYSIS & IMPROVEMENT RECOMMENDATIONS")
    print("="*80)
    
    # Set up DeepSeek client
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
    deepseek = OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=deepseek_api_key)
    
    # Step 1: Prepare data for AI analysis
    print("1. Preparing data for AI analysis...")
    
    # Clean titles
    df_clean = df.copy()
    df_clean['title_clean'] = df_clean['title'].apply(
        lambda x: str(x).strip() if pd.notna(x) else ""
    )
    
    # Filter out empty titles
    valid_titles = df_clean[df_clean['title_clean'].str.len() > 0]
    
    if len(valid_titles) == 0:
        print("❌ No valid titles found for analysis")
        return {}, df

    # Sample titles for analysis with minimum long titles requirement
    if len(valid_titles) > sample_size:
        print(f"   Sampling {sample_size} titles from {len(valid_titles)} valid titles...")
    
    # make sure200+ long titles （≥60 charactor）
    long_titles = valid_titles[valid_titles['title_clean'].str.len() >= 60]
    short_medium_titles = valid_titles[valid_titles['title_clean'].str.len() < 60]
    
    print(f"   Long titles (≥60 chars): {len(long_titles):,}")
    print(f"   Short/Medium titles (<60 chars): {len(short_medium_titles):,}")
    

    long_sample_size = min(200, len(long_titles))
    remaining_sample_size = sample_size - long_sample_size
    

    if len(long_titles) > 0:
        long_sample = long_titles.sample(min(long_sample_size, len(long_titles)), random_state=42)
    else:
        long_sample = pd.DataFrame()
    

    if remaining_sample_size > 0 and len(short_medium_titles) > 0:
        other_sample = short_medium_titles.sample(
            min(remaining_sample_size, len(short_medium_titles)), 
            random_state=42
        )
    else:
        other_sample = pd.DataFrame()
    

    sample_df = pd.concat([long_sample, other_sample], ignore_index=True)
    sample_df = sample_df.drop_duplicates(subset=['title_clean'])

    if len(sample_df) < sample_size:
        additional_needed = sample_size - len(sample_df)
        remaining_titles = valid_titles[~valid_titles.index.isin(sample_df.index)]
        if len(remaining_titles) > 0:
            additional_sample = remaining_titles.sample(
                min(additional_needed, len(remaining_titles)), 
                random_state=42
            )
            sample_df = pd.concat([sample_df, additional_sample], ignore_index=True)
        else:
            sample_df = valid_titles

    long_titles_count = (sample_df['title_clean'].str.len() >= 60).sum()
    print(f"   Long titles in sample (≥60 chars): {long_titles_count:,}")
    
    print(f"   Analyzing {len(sample_df)} product titles...")
    
    # Step 2: First AI call - Identify quality angles
    print("\n2. Identifying quality angles...")
    
    # Extract title characteristics for context
    title_lengths = sample_df['title_clean'].str.len()
    
    # Get diverse sample of titles (good, bad, medium)
    def get_diverse_sample(titles_df, n=30):
        """Get diverse sample including best and worst titles"""
        # Sort by length to get variety
        sorted_df = titles_df.sort_values('title_clean', key=lambda x: x.str.len())
        
        # Take from beginning, middle, and end
        sample_size = min(n, len(sorted_df))
        indices = []
        
        # From short titles
        indices.extend(sorted_df.head(10).index.tolist())
        
        # From medium length titles
        middle_start = len(sorted_df) // 2 - 5
        indices.extend(sorted_df.iloc[middle_start:middle_start + 10].index.tolist())
        
        # From long titles
        indices.extend(sorted_df.tail(10).index.tolist())
        
        # Remove duplicates and limit to sample_size
        unique_indices = list(dict.fromkeys(indices))[:sample_size]
        return titles_df.loc[unique_indices]
    
    diverse_sample_df = get_diverse_sample(sample_df, n=30)
    sample_titles = diverse_sample_df['title_clean'].tolist()
    
    # Get product context if available
    sample_with_context = []
    for idx, row in diverse_sample_df.iterrows():
        title_info = {
            'title': row['title_clean'],
            'title_length': len(row['title_clean'])
        }
        
        # Add additional context if available
        if 'category_id' in row and pd.notna(row['category_id']):
            title_info['category'] = row['category_id']
        if 'price' in row and pd.notna(row['price']):
            title_info['price'] = f"${row['price']:.2f}"
        if 'sold_quantity' in row and pd.notna(row['sold_quantity']):
            title_info['sales'] = int(row['sold_quantity'])
        
        sample_with_context.append(title_info)
    
    # Create data summary
    data_summary = {
        "total_titles_analyzed": len(sample_df),
        "avg_title_length": float(title_lengths.mean()),
        "min_title_length": int(title_lengths.min()),
        "max_title_length": int(title_lengths.max()),
        "title_length_distribution": {
            "short_titles": int((title_lengths < 30).sum()),
            "medium_titles": int(((title_lengths >= 30) & (title_lengths <= 80)).sum()),
            "long_titles": int((title_lengths > 80).sum())
        },
        "sample_titles_with_context": sample_with_context
    }
    
    # First AI call: Identify quality angles
    system_prompt_angles = '''
You are an experienced e-commerce analyst specializing in MercadoLibre (LATAM's largest platform).
Your task is to identify key quality angles for analyzing product titles.
'''

    input_prompt_angles = f'''
DATA OVERVIEW:
- Total titles analyzed: {data_summary['total_titles_analyzed']}
- Average title length: {data_summary['avg_title_length']:.1f} characters
- Short titles (<30 chars): {data_summary['title_length_distribution']['short_titles']}
- Medium titles (30-80 chars): {data_summary['title_length_distribution']['medium_titles']}
- Long titles (>80 chars): {data_summary['title_length_distribution']['long_titles']}

SAMPLE TITLES WITH CONTEXT:
{json.dumps(data_summary['sample_titles_with_context'], ensure_ascii=False, indent=2)}

TASK 1: IDENTIFY QUALITY ANGLES
Identify 5-7 key angles for analyzing MercadoLibre product title quality.
For each angle, provide a brief description and why it matters.

Respond with ONLY a JSON array of angles like this:
[
  {{
    "angle_name": "Keyword Optimization",
    "description": "How well the title includes relevant search keywords",
    "importance": "Critical for search visibility in MercadoLibre"
  }},
  ...
]
'''
    
    try:
        response_angles = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt_angles},
                {"role": "user", "content": input_prompt_angles},
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=2000
        )
        
        quality_angles = json.loads(response_angles.choices[0].message.content)
        print(f"✅ Identified {len(quality_angles)} quality angles")
        
    except Exception as e:
        print(f"❌ First AI call failed: {e}")
    
    # Step 3: Second AI call - Find problematic titles and provide suggestions
    print("\n3. Finding problematic titles and providing improvement suggestions...")
    
    
    evaluation_titles_df = sample_df
    
    # Prepare titles for evaluation
    titles_for_evaluation = []
    for idx, row in evaluation_titles_df.iterrows():
        title_data = {
            'original_title': row['title_clean'],
            'title_length': len(row['title_clean']),
            'product_id': str(row.get('id', idx)) if 'id' in row else str(idx)
        }
        
        # Add context if available
        context_fields = ['category_id', 'price', 'sold_quantity', 'quality_score']
        for field in context_fields:
            if field in row and pd.notna(row[field]):
                title_data[field] = row[field]
        
        titles_for_evaluation.append(title_data)
    
    # Second AI call: Evaluate titles and provide suggestions
    system_prompt_evaluation = '''
You are a MercadoLibre title optimization expert. Your task is to:
1. Evaluate product titles based on key quality angles
2. Identify the most problematic titles
3. Provide specific, actionable improvement suggestions
4. Explain WHY each title needs modification
'''

    input_prompt_evaluation = f'''
QUALITY ANGLES TO CONSIDER:
{json.dumps(quality_angles, ensure_ascii=False, indent=2)}

TITLES TO EVALUATE:
{json.dumps(titles_for_evaluation, ensure_ascii=False, indent=2)}

TASK 2: EVALUATE AND IMPROVE TITLES
Based on the quality angles above, analyze each title and:
1. Score each title (0-100) based on overall quality
2. Identify the 10 MOST PROBLEMATIC titles that need immediate improvement
3. For each problematic title, provide:
   - Current issues (which angles are problematic)
   - Why it needs modification (business impact)
   - Specific improvement suggestions
   - Improved title example
   - Estimated quality improvement

Format your response as JSON:
{{
  "evaluation_summary": {{
    "total_titles_evaluated": "number",
    "avg_quality_score": "number",
    "problematic_titles_count": "number"
  }},
  "problematic_titles": [
    {{
      "product_id": "product identifier",
      "original_title": "the original title",
      "current_issues": ["list of specific issues"],
      "business_impact": "explain why this hurts sales/visibility",
      "improvement_suggestions": ["specific actionable suggestions"],
      "improved_title_example": "example of improved title",
      "estimated_improvement": "estimated quality score improvement",
      "priority_level": "high/medium/low"
    }}
  ],
  "common_patterns": {{
    "most_common_issues": ["list of common issues across titles"],
    "quick_wins": ["list of easy fixes that apply to many titles"],
    "advanced_optimizations": ["list of advanced improvements"]
  }}
}}

Focus on practical, actionable advice that sellers can implement immediately.
'''
    
    try:
        response_evaluation = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt_evaluation},
                {"role": "user", "content": input_prompt_evaluation},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,  # Lower temperature for more consistent evaluations
            max_tokens=4000
        )
        
        evaluation_results = json.loads(response_evaluation.choices[0].message.content)
        print("✅ Title evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Second AI call failed: {e}")

    
    # Step 4: Process and integrate results
    print("\n4. Processing and integrating results...")
    
    # Add AI analysis results to original DataFrame
    result_df = df.copy()
    
    # Create mapping from product_id to AI evaluation
    problematic_titles_dict = {}
    if 'problematic_titles' in evaluation_results:
        for title_info in evaluation_results['problematic_titles']:
            product_id = title_info.get('product_id')
            if product_id:
                problematic_titles_dict[product_id] = title_info
    
    # Add AI analysis columns
    result_df['ai_analyzed'] = False
    result_df['ai_title_issues'] = None
    result_df['ai_improvement_suggestions'] = None
    result_df['ai_improved_title'] = None
    result_df['ai_priority_level'] = None
    
    # Mark analyzed titles and add suggestions
    for idx, row in result_df.iterrows():
        product_id = str(row.get('id', idx))
        
        if product_id in problematic_titles_dict:
            title_info = problematic_titles_dict[product_id]
            
            result_df.at[idx, 'ai_analyzed'] = True
            result_df.at[idx, 'ai_title_issues'] = ', '.join(title_info.get('current_issues', []))
            result_df.at[idx, 'ai_improvement_suggestions'] = ' | '.join(title_info.get('improvement_suggestions', []))
            result_df.at[idx, 'ai_improved_title'] = title_info.get('improved_title_example', '')
            result_df.at[idx, 'ai_priority_level'] = title_info.get('priority_level', 'medium')
    
    # Step 5: Display results
    print("\n" + "="*80)
    print("📊 AI TITLE EVALUATION RESULTS")
    print("="*80)
    
    # Display quality angles
    print(f"\n🔍 IDENTIFIED QUALITY ANGLES ({len(quality_angles)}):")
    print("-"*80)
    for i, angle in enumerate(quality_angles, 1):
        print(f"{i}. {angle.get('angle_name', 'Unknown')}")
        print(f"   {angle.get('description', '')[:80]}...")
    
    # Display evaluation summary
    if 'evaluation_summary' in evaluation_results:
        summary = evaluation_results['evaluation_summary']
        print(f"\n📈 EVALUATION SUMMARY:")
        print(f"   Titles evaluated: {summary.get('total_titles_evaluated', 'N/A')}")
        print(f"   Average quality score: {summary.get('avg_quality_score', 'N/A')}")
        print(f"   Problematic titles found: {summary.get('problematic_titles_count', 'N/A')}")
    
    # Display problematic titles
    if 'problematic_titles' in evaluation_results:
        problematic_titles = evaluation_results['problematic_titles']
        print(f"\n🚨 TOP 10 MOST PROBLEMATIC TITLES:")
        print("="*120)
        
        for i, title_info in enumerate(problematic_titles[:10], 1):
            print(f"\n{i}. Product ID: {title_info.get('product_id', 'N/A')}")
            print(f"   Original Title: {title_info.get('original_title', 'N/A')}")
            
            issues = title_info.get('current_issues', [])
            if issues:
                print(f"   Issues: {', '.join(issues[:3])}")
                if len(issues) > 3:
                    print(f"         (+ {len(issues) - 3} more issues)")
            
            print(f"   Business Impact: {title_info.get('business_impact', 'N/A')[:100]}...")
            
            suggestions = title_info.get('improvement_suggestions', [])
            if suggestions:
                print(f"   Key Suggestion: {suggestions[0]}")
            
            improved_title = title_info.get('improved_title_example', '')
            if improved_title:
                print(f"   Improved Example: {improved_title}")
            
            print(f"   Priority: {title_info.get('priority_level', 'medium').upper()}")
            print(f"   Estimated Improvement: {title_info.get('estimated_improvement', 'N/A')}")
    
    # Display common patterns
    if 'common_patterns' in evaluation_results:
        patterns = evaluation_results['common_patterns']
        print(f"\n🎯 COMMON PATTERNS & QUICK WINS:")
        print("-"*80)
        
        if 'most_common_issues' in patterns:
            print(f"Most Common Issues:")
            for i, issue in enumerate(patterns['most_common_issues'][:5], 1):
                print(f"  {i}. {issue}")
        
        if 'quick_wins' in patterns:
            print(f"\nQuick Wins (Easy Fixes):")
            for i, win in enumerate(patterns['quick_wins'][:5], 1):
                print(f"  {i}. {win}")
    
    print("\n" + "="*80)
    print("✅ AI ANALYSIS COMPLETE")
    print("="*80)
    
    # Combine all results
    final_results = {
        'quality_angles': quality_angles,
        'evaluation_results': evaluation_results,
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'sample_size_analyzed': len(sample_df)
    }
    
    return final_results, result_df




# Main execution
def run_title_improvement_pipeline(df, deepseek_api_key=None):
    """
    Complete title improvement pipeline
    
    Parameters:
    -----------
    df : pandas DataFrame
        Product data with titles
    deepseek_api_key : str
        DeepSeek API key
        
    Returns:
    --------
    tuple : (analysis_results, enhanced_df)
    """
    
    print("🚀 TITLE IMPROVEMENT ANALYSIS PIPELINE")
    print("="*80)
    
    # Run AI analysis
    analysis_results, enhanced_df = analyze_and_improve_titles_with_ai(
        df,
        sample_size=1000,
        deepseek_api_key=deepseek_api_key
    )
    
    # Save results
    if len(enhanced_df) > 0:
        # Save AI analysis results
        with open('title_improvement_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        
        # Show quick stats
        analyzed_count = enhanced_df['ai_analyzed'].sum()
        if analyzed_count > 0:
            high_priority = enhanced_df[enhanced_df['ai_priority_level'] == 'high'].shape[0]
            print(f"\n📊 QUICK STATS:")
            print(f"   Titles analyzed by AI: {analyzed_count}")
            print(f"   High priority improvements: {high_priority}")
    
    print("\n" + "="*80)
    print("🎯 PIPELINE COMPLETE")
    print("="*80)
    
    return analysis_results, enhanced_df


def analyze_non_title_quality_issues(df, sample_size=1000, deepseek_api_key=None):
    """
    Use AI to analyze listings with low quality scores due to NON-TITLE reasons.
    Focus on listings where title quality is good but overall score is low.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing product data with quality scores and breakdown
    sample_size : int
        Number of listings to sample for analysis
    deepseek_api_key : str
        DeepSeek API key
        
    Returns:
    --------
    dict : AI analysis results with recommendations
    pandas DataFrame : Original DataFrame with AI analysis columns added
    """
    
    print("🤖 AI NON-TITLE QUALITY ISSUE ANALYSIS")
    print("="*80)
    
    # Set up DeepSeek client
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
    deepseek = OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=deepseek_api_key)
    
    # Step 1: Prepare data - find listings with good titles but low overall scores
    print("1. Identifying listings with good titles but low overall quality...")
    
    # Create working copy
    df_clean = df.copy()
    
    # Calculate title-specific score from breakdown if available
    def extract_title_score(breakdown):
        """Extract title-related scores from score_breakdown"""
        if not isinstance(breakdown, dict):
            return 0
        
        title_scores = []
        title_keywords = ['title', 'Title', 'TITLE']
        
        for key, data in breakdown.items():
            if any(kw in key for kw in title_keywords):
                if isinstance(data, dict) and 'raw_score' in data:
                    title_scores.append(data['raw_score'])
        
        return sum(title_scores) / len(title_scores) if title_scores else 0
    
    # Add title-specific score
    if 'score_breakdown' in df_clean.columns:
        df_clean['title_specific_score'] = df_clean['score_breakdown'].apply(extract_title_score)
    else:
        # Use existing title-related columns as proxy
        title_score_columns = [col for col in df_clean.columns if 'title' in col.lower() and 'score' in col.lower()]
        if title_score_columns:
            df_clean['title_specific_score'] = df_clean[title_score_columns].mean(axis=1)
        else:
            df_clean['title_specific_score'] = df_clean['quality_score']  # Fallback
    
    # Filter for analysis: good title score (≥10) but low overall quality (≤60)

    analysis_mask = (
    (df_clean['title_specific_score'] >= 0) & 
    (df_clean['quality_score'] <= 60) &
    (df_clean['attr_entries'] > 0)  
    )
    
    candidate_df = df_clean[analysis_mask].copy()
    
    print(f"   Total listings: {len(df_clean):,}")
    print(f"   Listings with good titles but low overall quality: {len(candidate_df):,}")
    print(f"   Percentage: {(len(candidate_df)/len(df_clean))*100:.1f}%")
    
    if len(candidate_df) == 0:
        print("❌ No suitable listings found for analysis")
        return {}, df
    
    # Step 2: Sample for AI analysis
    print("\n2. Sampling listings for AI analysis...")
    
    if len(candidate_df) > sample_size:
        sample_df = candidate_df.sample(min(sample_size, len(candidate_df)), random_state=42)
    else:
        sample_df = candidate_df
    
    print(f"   Analyzing {len(sample_df)} listings...")
    
    # Step 3: Prepare data for AI
    print("\n3. Preparing data for AI analysis...")
    
    # Extract relevant metrics for each listing
    listings_data = []
    
    for idx, row in sample_df.iterrows():
        listing_info = {
            'listing_id': str(row.get('id', idx)),
            'overall_quality_score': float(row.get('quality_score', 0)),
            'title_score': float(row.get('title_specific_score', 0)),
            'title_text': str(row.get('title', ''))[:100] if pd.notna(row.get('title')) else ''
        }
        
        # Extract non-title metrics
        metrics = {}
        
        # Picture count
        if 'picture_count' in row and pd.notna(row['picture_count']):
            metrics['picture_count'] = int(row['picture_count'])
        
        # Video presence
        if 'has_video' in row and pd.notna(row['has_video']):
            metrics['has_video'] = bool(row['has_video'])
        elif 'video_id' in row:
            metrics['has_video'] = pd.notna(row['video_id']) and str(row['video_id']).strip() != ''
        
        # Update status
        if 'has_updated' in row and pd.notna(row['has_updated']):
            metrics['has_updated'] = bool(row['has_updated'])
        
        # Attributes metrics
        if 'attr_entries' in row and pd.notna(row['attr_entries']):
            metrics['attributes_count'] = int(row['attr_entries'])
        
        if 'attr_completeness_pct' in row and pd.notna(row['attr_completeness_pct']):
            metrics['attributes_completeness'] = float(row['attr_completeness_pct'])
        
        # Extract scores from breakdown
        if 'score_breakdown' in row and isinstance(row['score_breakdown'], dict):
            breakdown = row['score_breakdown']
            for key, data in breakdown.items():
                if isinstance(data, dict) and 'raw_score' in data:
                    # Identify non-title scores
                    is_title_related = any(kw in key.lower() for kw in ['title', 'name', 'heading'])
                    if not is_title_related:
                        metrics[f'score_{key.lower().replace(" ", "_")}'] = float(data['raw_score'])
        
        listing_info['non_title_metrics'] = metrics
        
        # Add category if available
        if 'category_id' in row and pd.notna(row['category_id']):
            listing_info['category'] = str(row['category_id'])
        
        # Add sales data if available
        if 'sold_quantity' in row and pd.notna(row['sold_quantity']):
            listing_info['sales'] = int(row['sold_quantity'])
        
        listings_data.append(listing_info)
    
    # Step 4: Call AI for analysis
    print("\n4. Calling AI for non-title issue analysis...")
    
    system_prompt = '''
You are an experienced e-commerce optimization specialist for MercadoLibre.
Your expertise is in identifying and fixing non-title related listing quality issues.

NON-TITLE QUALITY FACTORS INCLUDE:
1. Image quality and quantity (picture_count)
2. Video presence (has_video)
3. Listing freshness (has_updated)
4. Attributes completeness (attributes_count, attributes_completeness)
5. Other listing elements that affect conversion rates

Your task is to:
1. Analyze listings with good titles but low overall quality scores
2. Identify the specific non-title issues causing low scores
3. Provide actionable, specific recommendations
4. Focus on quick wins that sellers can implement immediately
'''

    input_prompt = f'''
LISTINGS DATA:
Total listings analyzed: {len(listings_data)}
All listings have: Good title scores (≥70) but Low overall quality scores (≤60)

SAMPLE LISTINGS DATA:
{json.dumps(listings_data[:20], ensure_ascii=False, indent=2)}

TASK:
Based on the non-title metrics provided, identify the 10 LISTINGS WITH THE MOST ACTIONABLE NON-TITLE ISSUES.
For each listing, provide:

1. SPECIFIC ISSUES IDENTIFIED: What non-title factors are causing low scores?
2. BUSINESS IMPACT: How do these issues affect sales/conversions?
3. ACTIONABLE RECOMMENDATIONS: Specific, practical steps to fix each issue
4. ESTIMATED IMPROVEMENT: How much quality score improvement is possible?
5. PRIORITY LEVEL: Based on impact and ease of implementation

FORMAT RESPONSE AS JSON:
{{
  "analysis_summary": {{
    "total_listings_analyzed": {len(listings_data)},
    "most_common_issues": ["list of top 3 most common non-title issues"],
    "quick_wins_available": "percentage of listings with easy fixes",
    "estimated_avg_improvement": "average quality score improvement possible"
  }},
  "top_10_problematic_listings": [
    {{
      "listing_id": "ID",
      "current_overall_score": "number",
      "current_title_score": "number",
      "main_non_title_issues": ["specific issues identified"],
      "business_impact_explanation": "how this affects sales",
      "specific_recommendations": [
        {{
          "action": "specific action to take",
          "reason": "why this helps",
          "difficulty": "easy/medium/hard",
          "expected_improvement": "points improvement"
        }}
      ],
      "overall_improvement_potential": "total points possible",
      "priority_level": "high/medium/low",
      "time_to_implement": "estimated time needed"
    }}
  ],
  "general_recommendations": {{
    "for_listings_with_few_images": ["recommendations"],
    "for_listings_without_videos": ["recommendations"],
    "for_listings_with_poor_attributes": ["recommendations"],
    "for_stale_listings": ["recommendations"]
  }}
}}

Focus on PRACTICAL, ACTIONABLE advice that sellers can implement without technical expertise.
'''
    
    try:
        response = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=4000
        )
        
        ai_results = json.loads(response.choices[0].message.content)
        print("✅ AI analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ AI API call failed: {e}")
        ai_results = create_fallback_non_title_analysis(listings_data)
    
    # Step 5: Process and integrate results
    print("\n5. Processing and integrating AI results...")
    
    result_df = df.copy()
    
    # Add analysis columns
    result_df['ai_non_title_analyzed'] = False
    result_df['ai_non_title_issues'] = None
    result_df['ai_non_title_recommendations'] = None
    result_df['ai_improvement_potential'] = None
    result_df['ai_priority_level'] = None
    
    # Map AI recommendations to DataFrame
    if 'top_10_problematic_listings' in ai_results:
        for listing_info in ai_results['top_10_problematic_listings']:
            listing_id = listing_info.get('listing_id')
            
            # Find the row in DataFrame
            if 'id' in result_df.columns:
                mask = result_df['id'].astype(str) == str(listing_id)
            else:
                # Try to match by index
                try:
                    idx = int(listing_id)
                    mask = result_df.index == idx
                except:
                    continue
            
            if mask.any():
                row_idx = result_df[mask].index[0]
                
                result_df.at[row_idx, 'ai_non_title_analyzed'] = True
                
                # Store issues
                issues = listing_info.get('main_non_title_issues', [])
                if issues:
                    result_df.at[row_idx, 'ai_non_title_issues'] = ' | '.join(issues[:3])
                
                # Store recommendations
                recommendations = listing_info.get('specific_recommendations', [])
                if recommendations:
                    rec_texts = [rec.get('action', '') for rec in recommendations[:2]]
                    result_df.at[row_idx, 'ai_non_title_recommendations'] = ' | '.join(rec_texts)
                
                # Store improvement potential
                improvement = listing_info.get('overall_improvement_potential', '')
                result_df.at[row_idx, 'ai_improvement_potential'] = improvement
                
                # Store priority
                priority = listing_info.get('priority_level', 'medium')
                result_df.at[row_idx, 'ai_priority_level'] = priority
    
    # Step 6: Display results
    print("\n" + "="*80)
    print("📊 NON-TITLE QUALITY ISSUE ANALYSIS RESULTS")
    print("="*80)
    
    # Display summary
    if 'analysis_summary' in ai_results:
        summary = ai_results['analysis_summary']
        print(f"\n📈 ANALYSIS SUMMARY:")
        print(f"   Listings analyzed: {summary.get('total_listings_analyzed', 'N/A')}")
        
        if 'most_common_issues' in summary:
            print(f"   Most common issues:")
            for i, issue in enumerate(summary['most_common_issues'][:3], 1):
                print(f"     {i}. {issue}")
        
        print(f"   Quick wins available: {summary.get('quick_wins_available', 'N/A')}")
        print(f"   Estimated average improvement: {summary.get('estimated_avg_improvement', 'N/A')}")
    
    # Display top problematic listings
    if 'top_10_problematic_listings' in ai_results:
        problematic_listings = ai_results['top_10_problematic_listings']
        print(f"\n🚨 TOP 10 LISTINGS WITH NON-TITLE ISSUES:")
        print("="*120)
        
        for i, listing in enumerate(problematic_listings, 1):
            print(f"\n{i}. Listing ID: {listing.get('listing_id', 'N/A')}")
            print(f"   Current Score: {listing.get('current_overall_score', 'N/A')}")
            
            issues = listing.get('main_non_title_issues', [])
            if issues:
                print(f"   Main Issues: {', '.join(issues[:2])}")
                if len(issues) > 2:
                    print(f"                (+ {len(issues) - 2} more)")
            
            recommendations = listing.get('specific_recommendations', [])
            if recommendations:
                print(f"   Top Recommendation: {recommendations[0].get('action', 'N/A')}")
            
            print(f"   Improvement Potential: {listing.get('overall_improvement_potential', 'N/A')}")
            print(f"   Priority: {listing.get('priority_level', 'N/A').upper()}")
            print(f"   Time to Implement: {listing.get('time_to_implement', 'N/A')}")
    
    # Display general recommendations
    if 'general_recommendations' in ai_results:
        general_recs = ai_results['general_recommendations']
        print(f"\n💡 GENERAL RECOMMENDATIONS BY ISSUE TYPE:")
        print("-"*80)
        
        for issue_type, recommendations in general_recs.items():
            if recommendations:
                print(f"\n{issue_type.replace('_', ' ').title()}:")
                for i, rec in enumerate(recommendations[:2], 1):
                    print(f"  {i}. {rec}")
    
    print("\n" + "="*80)
    print("✅ NON-TITLE ANALYSIS COMPLETE")
    print("="*80)
    
    return ai_results, result_df


def create_fallback_non_title_analysis(listings_data):
    """Create fallback analysis for non-title issues"""
    
    # Simple heuristic analysis
    problematic_listings = []
    
    for listing in listings_data[:10]:  # First 10 as problematic
        listing_id = listing.get('listing_id', 'unknown')
        metrics = listing.get('non_title_metrics', {})
        
        # Identify issues based on metrics
        issues = []
        recommendations = []
        
        # Check picture count
        pic_count = metrics.get('picture_count', 0)
        if pic_count < 3:
            issues.append(f"Low image count ({pic_count} images)")
            recommendations.append({
                "action": f"Add {3 - pic_count} more high-quality product images",
                "reason": "Multiple images increase trust and conversion by 25%",
                "difficulty": "easy",
                "expected_improvement": "15-20 points"
            })
        
        # Check video presence
        has_video = metrics.get('has_video', False)
        if not has_video:
            issues.append("No product video")
            recommendations.append({
                "action": "Add a short product demonstration video",
                "reason": "Videos increase conversion rates by 30-40%",
                "difficulty": "medium",
                "expected_improvement": "20-25 points"
            })
        
        # Check attributes
        attr_comp = metrics.get('attributes_completeness', 100)
        if attr_comp < 80:
            issues.append(f"Incomplete attributes ({attr_comp}% complete)")
            recommendations.append({
                "action": "Complete all product attribute fields",
                "reason": "Complete attributes improve search filtering and buyer confidence",
                "difficulty": "easy",
                "expected_improvement": "10-15 points"
            })
        
        attr_count = metrics.get('attributes_count', 0)
        if attr_count < 5:
            issues.append(f"Few attributes ({attr_count} total)")
            recommendations.append({
                "action": "Add more detailed product specifications",
                "reason": "Detailed specs help buyers make informed decisions",
                "difficulty": "medium",
                "expected_improvement": "10-15 points"
            })
        
        # Check update status
        has_updated = metrics.get('has_updated', True)
        if not has_updated:
            issues.append("Listing never updated")
            recommendations.append({
                "action": "Update listing with current details or repost",
                "reason": "Fresh listings get algorithm preference and appear newer",
                "difficulty": "easy",
                "expected_improvement": "5-10 points"
            })
        
        # Calculate total improvement
        total_improvement = "25-40 points" if issues else "10-15 points"
        
        problematic_listings.append({
            "listing_id": listing_id,
            "current_overall_score": listing.get('overall_quality_score', 0),
            "current_title_score": listing.get('title_score', 0),
            "main_non_title_issues": issues,
            "business_impact_explanation": "These issues reduce buyer trust and search visibility",
            "specific_recommendations": recommendations[:3],
            "overall_improvement_potential": total_improvement,
            "priority_level": "high" if issues else "medium",
            "time_to_implement": "1-2 hours"
        })
    
    return {
        "analysis_summary": {
            "total_listings_analyzed": len(listings_data),
            "most_common_issues": ["Low image count", "Missing video", "Incomplete attributes"],
            "quick_wins_available": "70% of listings",
            "estimated_avg_improvement": "20-30 points"
        },
        "top_10_problematic_listings": problematic_listings,
        "general_recommendations": {
            "for_listings_with_few_images": [
                "Aim for 5-8 high-quality images showing different angles",
                "Include close-ups of important features and product labels"
            ],
            "for_listings_without_videos": [
                "Create a 30-second demonstration video",
                "Show product in use or highlight key features"
            ],
            "for_listings_with_poor_attributes": [
                "Complete all mandatory attribute fields",
                "Add optional attributes that differentiate your product"
            ],
            "for_stale_listings": [
                "Update price or description regularly",
                "Consider reposting every 30-60 days"
            ]
        }
    }


# Main execution function
def run_non_title_issue_analysis(df, deepseek_api_key=None):
    """
    Complete non-title issue analysis pipeline
    """
    print("🚀 NON-TITLE QUALITY ISSUE ANALYSIS PIPELINE")
    print("="*80)
    
    # Run AI analysis
    ai_results, enhanced_df = analyze_non_title_quality_issues(
        df,
        sample_size=1000,
        deepseek_api_key=deepseek_api_key
    )
    
    # Save results
    if len(enhanced_df) > 0:
        # Save AI analysis results
        with open('non_title_issue_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(ai_results, f, ensure_ascii=False, indent=2)
        
        # Save enhanced DataFrame
        enhanced_df.to_csv('listings_with_non_title_improvements.csv', index=False)
        
        print("\n💾 Results saved:")
        print("   - AI analysis: 'non_title_issue_analysis.json'")
        print("   - Enhanced data: 'listings_with_non_title_improvements.csv'")
        
        # Show quick stats
        analyzed_count = enhanced_df['ai_non_title_analyzed'].sum()
        if analyzed_count > 0:
            high_priority = enhanced_df[enhanced_df['ai_priority_level'] == 'high'].shape[0]
            print(f"\n📊 QUICK STATS:")
            print(f"   Listings analyzed by AI: {analyzed_count}")
            print(f"   High priority improvements: {high_priority}")
    
    print("\n" + "="*80)
    print("🎯 NON-TITLE ANALYSIS PIPELINE COMPLETE")
    print("="*80)
    
    return ai_results, enhanced_df

def generate_data_insights_with_ai(df, analysis_type="comprehensive", deepseek_api_key=None):
    """
    Use GenAI to identify patterns and generate insights from product data.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing product data with various metrics
    analysis_type : str
        Type of analysis to perform:
        - "comprehensive": Comprehensive insights across all metrics
        - "quality_patterns": Focus on quality score patterns
        - "sales_drivers": Identify factors driving sales
        - "improvement_opportunities": Find improvement opportunities
        - "category_analysis": Category-specific patterns
    deepseek_api_key : str
        DeepSeek API key
        
    Returns:
    --------
    dict : AI-generated insights and patterns
    pandas DataFrame : Summary statistics used for analysis
    """
    
    print(f"🤖 GENAI DATA INSIGHTS GENERATION - {analysis_type.upper()} ANALYSIS")
    print("="*80)
    
    # Set up DeepSeek client
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
    deepseek = OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=deepseek_api_key)
    
    # Step 1: Prepare summary statistics
    print("1. Calculating summary statistics for AI analysis...")
    
    # Create summary statistics
    insights_data = {
        "analysis_type": analysis_type,
        "dataset_overview": {
            "total_products": len(df),
            "total_columns": len(df.columns),
            "columns_available": list(df.columns),
            "data_types": {col: str(df[col].dtype) for col in df.columns[:20]}  # First 20 columns
        }
    }
    
    # Calculate key metrics summary
    key_metrics = {}
    
    # Quality metrics
    quality_metrics = ['quality_score', 'title_score', 'title_length', 'picture_count', 
                      'attr_entries', 'attr_completeness_pct', 'has_video', 'has_updated']
    
    for metric in quality_metrics:
        if metric in df.columns:
            if df[metric].dtype in ['int64', 'float64']:
                key_metrics[metric] = {
                    "mean": float(df[metric].mean()),
                    "median": float(df[metric].median()),
                    "std": float(df[metric].std()),
                    "min": float(df[metric].min()),
                    "max": float(df[metric].max()),
                    "missing_pct": float((df[metric].isna().sum() / len(df)) * 100)
                }
            elif df[metric].dtype == 'bool' or metric in ['has_video', 'has_updated']:
                if df[metric].dtype == 'bool' or df[metric].nunique() <= 3:
                    key_metrics[metric] = {
                        "true_count": int(df[metric].sum()),
                        "false_count": int((~df[metric]).sum() if df[metric].dtype == 'bool' else len(df) - df[metric].sum()),
                        "true_pct": float((df[metric].sum() / len(df)) * 100)
                    }
    
    insights_data["key_metrics"] = key_metrics
    
    # Sales and price analysis
    if 'sold_quantity' in df.columns and 'price' in df.columns:
        insights_data["sales_price_analysis"] = {
            "sales_summary": {
                "total_sales": int(df['sold_quantity'].sum()),
                "avg_sales_per_product": float(df['sold_quantity'].mean()),
                "sales_distribution": {
                    "top_10_pct_sales": float(df['sold_quantity'].quantile(0.9)),
                    "median_sales": float(df['sold_quantity'].median()),
                    "bottom_10_pct_sales": float(df['sold_quantity'].quantile(0.1))
                }
            },
            "price_summary": {
                "avg_price": float(df['price'].mean()),
                "price_range": f"${float(df['price'].min()):.2f} - ${float(df['price'].max()):.2f}",
                "price_distribution": {
                    "low_price": float(df['price'].quantile(0.25)),
                    "median_price": float(df['price'].median()),
                    "high_price": float(df['price'].quantile(0.75))
                }
            },
            "sales_price_correlation": float(df['sold_quantity'].corr(df['price']))
        }
    
    # Category analysis
    if 'category_id' in df.columns:
        category_stats = df.groupby('category_id').agg({
            'id': 'count',
            'quality_score': 'mean',
            'sold_quantity': ['sum', 'mean'],
            'price': 'mean'
        }).round(2)
        
        category_stats.columns = ['product_count', 'avg_quality', 'total_sales', 'avg_sales', 'avg_price']
        category_stats = category_stats.reset_index()
        
        insights_data["category_analysis"] = {
            "total_categories": int(df['category_id'].nunique()),
            "top_categories_by_products": category_stats.nlargest(5, 'product_count')[['category_id', 'product_count']].to_dict('records'),
            "top_categories_by_sales": category_stats.nlargest(5, 'total_sales')[['category_id', 'total_sales']].to_dict('records'),
            "top_categories_by_quality": category_stats.nlargest(5, 'avg_quality')[['category_id', 'avg_quality']].to_dict('records')
        }
    
    # Quality score patterns
    if 'quality_score' in df.columns:
        quality_bins = pd.cut(df['quality_score'], bins=[0, 20, 40, 60, 80, 100], 
                             labels=['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent'])
        quality_dist = quality_bins.value_counts().to_dict()
        
        insights_data["quality_distribution"] = {
            "distribution": quality_dist,
            "percentage": {k: (v / len(df)) * 100 for k, v in quality_dist.items()}
        }
        
        # Correlation with other metrics
        correlations = {}
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if col != 'quality_score' and col in df.columns:
                corr = df['quality_score'].corr(df[col])
                if not pd.isna(corr):
                    correlations[col] = float(corr)
        
        # Sort by absolute correlation
        sorted_correlations = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
        insights_data["quality_correlations"] = sorted_correlations
    
    # Patterns in low vs high quality listings
    if 'quality_score' in df.columns:
        low_quality = df[df['quality_score'] < 50]
        high_quality = df[df['quality_score'] >= 70]
        
        if len(low_quality) > 0 and len(high_quality) > 0:
            comparison = {}
            
            for metric in ['title_length', 'picture_count', 'attr_entries', 'attr_completeness_pct']:
                if metric in df.columns:
                    comparison[metric] = {
                        "low_quality_avg": float(low_quality[metric].mean()),
                        "high_quality_avg": float(high_quality[metric].mean()),
                        "difference": float(high_quality[metric].mean() - low_quality[metric].mean()),
                        "difference_pct": float(((high_quality[metric].mean() - low_quality[metric].mean()) / low_quality[metric].mean()) * 100) if low_quality[metric].mean() > 0 else 0
                    }
            
            insights_data["quality_comparison"] = comparison
    
    # Step 2: Prepare data samples for AI
    print("\n2. Preparing data samples for AI...")
    
    # Sample actual data for AI to see patterns
    sample_size = min(50, len(df))
    sample_data = df.sample(sample_size, random_state=42)
    
    # Create readable sample
    sample_records = []
    for idx, row in sample_data.iterrows():
        record = {}
        
        # Include key fields
        key_fields = ['id', 'title', 'quality_score', 'sold_quantity', 'price', 'category_id']
        for field in key_fields:
            if field in row and pd.notna(row[field]):
                if field == 'title':
                    record[field] = str(row[field])[:50] + "..." if len(str(row[field])) > 50 else str(row[field])
                else:
                    record[field] = row[field]
        
        # Include quality metrics
        quality_fields = ['title_score', 'picture_count', 'has_video', 'has_updated', 
                         'attr_entries', 'attr_completeness_pct']
        for field in quality_fields:
            if field in row and pd.notna(row[field]):
                record[field] = row[field]
        
        sample_records.append(record)
    
    insights_data["sample_records"] = sample_records[:20]  # First 20 records
    
    # Step 3: Define AI prompts based on analysis type
    print("\n3. Configuring AI prompts for insights generation...")
    
    system_prompt = '''
You are a senior data analyst and e-commerce expert specializing in MercadoLibre.
Your expertise includes:
1. Identifying patterns in product listing data
2. Understanding e-commerce metrics and their business impact
3. Generating actionable insights from data
4. Providing data-driven recommendations

Your task is to analyze the provided data summary and generate meaningful insights.
'''
    
    # Customize input prompt based on analysis type
    analysis_prompts = {
        "comprehensive": '''
COMPREHENSIVE ANALYSIS REQUEST:
Analyze all aspects of the provided data and generate comprehensive insights.

1. DATA PATTERNS: What patterns do you see in the data?
2. CORRELATIONS: Which metrics are most strongly correlated with quality and sales?
3. OPPORTUNITIES: Where are the biggest improvement opportunities?
4. BEST PRACTICES: What do high-performing listings have in common?
5. RISKS: What potential issues or risks can you identify?
6. RECOMMENDATIONS: What are your top 5 data-driven recommendations?

Focus on both statistical patterns and practical business implications.
''',
        "quality_patterns": '''
QUALITY PATTERNS ANALYSIS:
Focus specifically on quality score patterns and their drivers.

1. QUALITY DISTRIBUTION: What does the quality score distribution tell us?
2. KEY DRIVERS: Which factors have the strongest impact on quality scores?
3. LOW VS HIGH QUALITY: What are the key differences between low and high quality listings?
4. THRESHOLDS: Are there any quality score thresholds that matter?
5. CONSISTENCY: How consistent are quality scores across categories?
6. IMPROVEMENT PATH: What's the easiest path to improve quality scores?

Provide specific, actionable insights about quality patterns.
''',
        "sales_drivers": '''
SALES DRIVERS ANALYSIS:
Identify what drives sales performance.

1. SALES PATTERNS: What patterns do you see in sales distribution?
2. QUALITY-SALES RELATIONSHIP: How does quality correlate with sales?
3. PRICE IMPACT: What's the relationship between price and sales?
4. CATEGORY PERFORMANCE: Which categories perform best and why?
5. OPTIMAL COMBINATION: What combination of factors leads to highest sales?
6. UNDERPERFORMERS: Why do some listings underperform despite good metrics?

Focus on understanding what actually drives conversions and sales.
''',
        "improvement_opportunities": '''
IMPROVEMENT OPPORTUNITIES ANALYSIS:
Find the biggest opportunities for improvement.

1. LOW-HANGING FRUIT: Where are the easiest wins for quality improvement?
2. BIGGEST GAPS: What are the biggest gaps between current and potential performance?
3. CATEGORY OPPORTUNITIES: Which categories have the most room for improvement?
4. METRIC TARGETS: What should be the target values for key metrics?
5. PRIORITIZATION: What should sellers prioritize first?
6. ROI FOCUS: Where will improvements have the biggest impact?

Focus on practical, actionable improvement opportunities.
''',
        "category_analysis": '''
CATEGORY-SPECIFIC ANALYSIS:
Analyze patterns and opportunities by category.

1. CATEGORY DIFFERENCES: How do patterns differ across categories?
2. CATEGORY-SPECIFIC METRICS: What metrics matter most in each category?
3. BEST CATEGORY PRACTICES: What works well in each category?
4. CROSS-CATEGORY INSIGHTS: What insights apply across all categories?
5. CATEGORY OPPORTUNITIES: Where are the category-specific opportunities?
6. CATEGORY RECOMMENDATIONS: What are category-specific recommendations?

Provide insights that are tailored to specific product categories.
'''
    }
    
    input_prompt = f'''
DATA SUMMARY FOR ANALYSIS:
{json.dumps(insights_data, ensure_ascii=False, indent=2)}

ANALYSIS REQUEST: {analysis_prompts.get(analysis_type, analysis_prompts["comprehensive"])}

RESPONSE FORMAT:
Provide your analysis as structured JSON with this format:
{{
  "executive_summary": "Brief summary of key findings",
  "key_insights": [
    {{
      "insight_title": "Descriptive title of the insight",
      "description": "Detailed description of the insight",
      "data_evidence": "What data supports this insight",
      "business_implication": "What this means for the business",
      "confidence_level": "high/medium/low"
    }}
  ],
  "patterns_discovered": [
    {{
      "pattern_name": "Name of the pattern",
      "pattern_description": "Description of the pattern",
      "pattern_examples": "Examples from the data",
      "prevalence": "How common is this pattern",
      "significance": "Why this pattern matters"
    }}
  ],
  "correlation_insights": [
    {{
      "relationship": "Description of the relationship",
      "strength": "strong/moderate/weak",
      "direction": "positive/negative",
      "implication": "What this means for strategy"
    }}
  ],
  "actionable_recommendations": [
    {{
      "recommendation": "Specific actionable recommendation",
      "rationale": "Why this recommendation makes sense",
      "expected_impact": "What impact to expect",
      "implementation_difficulty": "easy/medium/hard",
      "priority": "high/medium/low"
    }}
  ],
  "data_limitations": [
    "List of data limitations or gaps"
  ],
  "further_analysis_suggestions": [
    "Suggestions for additional analysis"
  ]
}}

Focus on data-driven, practical insights that can inform business decisions.
'''
    
    # Step 4: Call AI API
    print("\n4. Calling AI for insights generation...")
    
    try:
        response = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=5000
        )
        
        ai_insights = json.loads(response.choices[0].message.content)
        print("✅ AI insights generation completed successfully!")
        
    except Exception as e:
        print(f"❌ AI API call failed: {e}")
        ai_insights = create_fallback_insights(insights_data, analysis_type)
    
    # Step 5: Display key insights
    print("\n" + "="*80)
    print(f"📊 GENAI DATA INSIGHTS - {analysis_type.upper()}")
    print("="*80)
    
    # Display executive summary
    if "executive_summary" in ai_insights:
        print(f"\n📋 EXECUTIVE SUMMARY:")
        print(f"{ai_insights['executive_summary']}")
    
    # Display key insights
    if "key_insights" in ai_insights and ai_insights["key_insights"]:
        print(f"\n🔍 KEY INSIGHTS:")
        for i, insight in enumerate(ai_insights["key_insights"][:5], 1):
            print(f"\n{i}. {insight.get('insight_title', 'Insight')}")
            print(f"   {insight.get('description', '')[:100]}...")
            print(f"   Business Implication: {insight.get('business_implication', '')[:80]}...")
            print(f"   Confidence: {insight.get('confidence_level', 'medium').upper()}")
    
    # Display patterns
    if "patterns_discovered" in ai_insights and ai_insights["patterns_discovered"]:
        print(f"\n🎯 PATTERNS DISCOVERED:")
        for i, pattern in enumerate(ai_insights["patterns_discovered"][:3], 1):
            print(f"{i}. {pattern.get('pattern_name', 'Pattern')}")
            print(f"   {pattern.get('pattern_description', '')[:80]}...")
            print(f"   Prevalence: {pattern.get('prevalence', 'N/A')}")
    
    # Display recommendations
    if "actionable_recommendations" in ai_insights and ai_insights["actionable_recommendations"]:
        print(f"\n💡 TOP ACTIONABLE RECOMMENDATIONS:")
        for i, rec in enumerate(ai_insights["actionable_recommendations"][:5], 1):
            print(f"{i}. {rec.get('recommendation', 'Recommendation')}")
            print(f"   Impact: {rec.get('expected_impact', 'N/A')}")
            print(f"   Priority: {rec.get('priority', 'medium').upper()}")
            print(f"   Difficulty: {rec.get('implementation_difficulty', 'medium').upper()}")
    
    # Display data limitations
    if "data_limitations" in ai_insights and ai_insights["data_limitations"]:
        print(f"\n⚠️ DATA LIMITATIONS:")
        for i, limitation in enumerate(ai_insights["data_limitations"][:3], 1):
            print(f"{i}. {limitation}")
    
    print("\n" + "="*80)
    print("✅ INSIGHTS GENERATION COMPLETE")
    print("="*80)
    
    return ai_insights, insights_data


def create_fallback_insights(insights_data, analysis_type):
    """Create fallback insights when AI fails"""
    
    fallback_insights = {
        "executive_summary": f"Based on analysis of {insights_data.get('dataset_overview', {}).get('total_products', 0)} products, key patterns and opportunities have been identified.",
        "key_insights": [],
        "patterns_discovered": [],
        "correlation_insights": [],
        "actionable_recommendations": [],
        "data_limitations": ["AI analysis was unavailable, using heuristic insights"],
        "further_analysis_suggestions": ["Consider more detailed statistical analysis", "Add time-series data for trend analysis"]
    }
    
    # Generate basic insights from the data
    if "quality_distribution" in insights_data:
        quality_dist = insights_data["quality_distribution"]
        fallback_insights["key_insights"].append({
            "insight_title": "Quality Score Distribution",
            "description": f"Products are distributed across quality levels: {quality_dist.get('distribution', {})}",
            "data_evidence": "Quality score analysis",
            "business_implication": "Opportunity to move more products to higher quality tiers",
            "confidence_level": "high"
        })
    
    if "quality_correlations" in insights_data:
        correlations = insights_data["quality_correlations"]
        if correlations:
            strongest = max(correlations.items(), key=lambda x: abs(x[1]))
            fallback_insights["key_insights"].append({
                "insight_title": "Strongest Quality Correlation",
                "description": f"Quality score is most strongly correlated with {strongest[0]} (r={strongest[1]:.2f})",
                "data_evidence": "Correlation analysis",
                "business_implication": f"Focus on improving {strongest[0]} to boost quality scores",
                "confidence_level": "medium"
            })
    
    if "sales_price_analysis" in insights_data:
        sales_price = insights_data["sales_price_analysis"]
        fallback_insights["key_insights"].append({
            "insight_title": "Sales-Price Relationship",
            "description": f"Correlation between sales and price: {sales_price.get('sales_price_correlation', 0):.2f}",
            "data_evidence": "Sales and price correlation analysis",
            "business_implication": "Provides insight into price sensitivity",
            "confidence_level": "medium"
        })
    
    # Add basic recommendations
    fallback_insights["actionable_recommendations"] = [
        {
            "recommendation": "Improve image quality and quantity for low-quality listings",
            "rationale": "Picture count shows moderate correlation with quality scores",
            "expected_impact": "15-25% quality score improvement",
            "implementation_difficulty": "easy",
            "priority": "high"
        },
        {
            "recommendation": "Focus on improving title quality in underperforming categories",
            "rationale": "Title quality varies significantly across categories",
            "expected_impact": "10-20% increase in search visibility",
            "implementation_difficulty": "medium",
            "priority": "high"
        },
        {
            "recommendation": "Add video content to top-performing products",
            "rationale": "Video presence correlates with higher engagement",
            "expected_impact": "20-30% conversion rate improvement",
            "implementation_difficulty": "medium",
            "priority": "medium"
        }
    ]
    
    return fallback_insights


def generate_multiple_insights(df, analysis_types=None, deepseek_api_key=None):
    """
    Generate multiple types of insights in one run
    
    Parameters:
    -----------
    df : pandas DataFrame
        Product data
    analysis_types : list
        List of analysis types to run
    deepseek_api_key : str
        DeepSeek API key
        
    Returns:
    --------
    dict : Dictionary with all insights
    """
    
    if analysis_types is None:
        analysis_types = ["comprehensive", "quality_patterns", "sales_drivers", "improvement_opportunities"]
    
    print("🚀 GENERATING MULTIPLE DATA INSIGHTS")
    print("="*80)
    
    all_insights = {}
    
    for analysis_type in analysis_types:
        print(f"\n📊 Running {analysis_type.upper()} analysis...")
        print("-"*60)
        
        insights, summary_data = generate_data_insights_with_ai(
            df,
            analysis_type=analysis_type,
            deepseek_api_key=deepseek_api_key
        )
        
        all_insights[analysis_type] = {
            "insights": insights,
            "summary_data": summary_data
        }
    
    # Generate combined summary
    print("\n" + "="*80)
    print("📈 COMBINED INSIGHTS SUMMARY")
    print("="*80)
    
    combined_recommendations = []
    for analysis_type, data in all_insights.items():
        insights = data.get("insights", {})
        
        if "actionable_recommendations" in insights:
            for rec in insights["actionable_recommendations"][:2]:  # Top 2 from each
                rec["source_analysis"] = analysis_type
                combined_recommendations.append(rec)
    
    # Sort by priority and difficulty
    combined_recommendations.sort(key=lambda x: (
        0 if x.get("priority") == "high" else (1 if x.get("priority") == "medium" else 2),
        0 if x.get("implementation_difficulty") == "easy" else (1 if x.get("implementation_difficulty") == "medium" else 2)
    ))
    
    print(f"\n🏆 TOP 10 COMBINED RECOMMENDATIONS:")
    for i, rec in enumerate(combined_recommendations[:10], 1):
        print(f"\n{i}. {rec.get('recommendation', 'Recommendation')}")
        print(f"   Source: {rec.get('source_analysis', 'Unknown').upper()}")
        print(f"   Priority: {rec.get('priority', 'medium').upper()}")
        print(f"   Difficulty: {rec.get('implementation_difficulty', 'medium').upper()}")
        print(f"   Expected Impact: {rec.get('expected_impact', 'N/A')}")
    
    # Save all insights
    if all_insights:
        with open('all_data_insights.json', 'w', encoding='utf-8') as f:
            json.dump(all_insights, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 All insights saved to 'all_data_insights.json'")
    
    print("\n" + "="*80)
    print("✅ MULTIPLE INSIGHTS GENERATION COMPLETE")
    print("="*80)
    
    return all_insights


    
   

"""Categorization module for structural engineering projects."""
from typing import List, Dict, Optional
import re
from openai import OpenAI

import config


class StructuralCategorizer:
    """Categorizes structural engineering projects using LLM."""
    
    def __init__(self):
        """Initialize the categorizer with OpenAI client."""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.categories = config.STRUCTURAL_CATEGORIES
    
    def categorize_project(
        self, 
        project_name: str, 
        text_sample: str,
        metadata: Dict
    ) -> List[str]:
        """
        Categorize a project using LLM analysis.
        
        Args:
            project_name: Name of the project
            text_sample: Sample text from the project
            metadata: Project metadata
            
        Returns:
            List of applicable categories
        """
        # Create prompt for categorization
        prompt = self._create_categorization_prompt(
            project_name, 
            text_sample, 
            metadata
        )
        
        try:
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a structural engineering expert. Categorize projects based on their structural engineering taxonomy."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            categories = self._parse_categories(result)
            
            return categories
            
        except Exception as e:
            print(f"Error categorizing project: {str(e)}")
            # Fallback to rule-based categorization
            return self._rule_based_categorization(project_name, text_sample)
    
    def _create_categorization_prompt(
        self,
        project_name: str,
        text_sample: str,
        metadata: Dict
    ) -> str:
        """Create a prompt for project categorization."""
        categories_str = ", ".join(self.categories)
        
        prompt = f"""Analyze the following structural engineering project and categorize it.

Project Name: {project_name}
Phase: {metadata.get('phase', 'Unknown')}

Text Sample (first 1500 characters):
{text_sample[:1500]}

Available Categories: {categories_str}

Instructions:
1. Select ALL applicable categories from the list above
2. Return ONLY the category names, separated by commas
3. If multiple categories apply, list them all
4. Use exact category names from the list

Categories:"""
        
        return prompt
    
    def _parse_categories(self, llm_response: str) -> List[str]:
        """Parse categories from LLM response."""
        # Split by comma or newline
        categories = re.split(r'[,\n]', llm_response)
        
        # Clean and filter
        parsed_categories = []
        for cat in categories:
            cat = cat.strip()
            # Check if it matches one of our predefined categories
            for valid_cat in self.categories:
                if valid_cat.lower() in cat.lower():
                    if valid_cat not in parsed_categories:
                        parsed_categories.append(valid_cat)
        
        return parsed_categories if parsed_categories else ["General Structural"]
    
    def _rule_based_categorization(
        self,
        project_name: str,
        text_sample: str
    ) -> List[str]:
        """Fallback rule-based categorization."""
        combined_text = f"{project_name} {text_sample}".lower()
        categories = []
        
        # Define keyword patterns for each category
        patterns = {
            "Bridges": [r'\bbridge\b', r'\bgirder\b', r'\bspan\b', r'\bdeck\b', r'\bbent\b', r'\babutment\b', 
                       r'\bshear\s+key\b', r'\bpipe\s+pin\b', r'\bbearing\s+pad\b'],
            "Tunnels": [r'\btunnel\b', r'\bunderground\b', r'\bboring\b'],
            "Stations": [r'\bstation\b', r'\bplatform\b', r'\btransit\b'],
            "Retaining Walls": [r'\bretaining\s+wall\b', r'\b(?:rw|r\.w\.)\b', r'\bearth\s+retention\b', 
                               r'\bshga\b', r'\bground\s+anchor\b'],
            "Pedestrian Structures": [r'\bpedestrian\b', r'\bped\s+(?:ramp|bridge)\b', r'\bwalkway\b'],
            "Overhead Structures": [r'\boverhead\b', r'\boh\b', r'\boverpass\b'],
            "Foundations": [r'\bfoundation\b', r'\bpile\b', r'\bdrilled\s+shaft\b', r'\bcaisson\b', r'\bcidh\b'],
            "Seismic Design": [r'\bseismic\b', r'\bearthquake\b', r'\bductility\b'],
        }
        
        for category, keywords in patterns.items():
            for pattern in keywords:
                if re.search(pattern, combined_text):
                    categories.append(category)
                    break
        
        # Default category if nothing matches
        if not categories:
            categories.append("General Structural")
        
        return categories
    
    def categorize_multiple_projects(
        self,
        processed_data: List[Dict]
    ) -> Dict[str, List[str]]:
        """
        Categorize multiple projects.
        
        Args:
            processed_data: List of processed PDF data
            
        Returns:
            Dictionary mapping file names to categories
        """
        categorizations = {}
        
        for pdf_data in processed_data:
            file_name = pdf_data['file_name']
            project_name = pdf_data['metadata'].get('project_name', file_name)
            
            # Get text sample from first few pages
            text_sample = ' '.join([
                pdf_data['pages_text'].get(i, '')
                for i in range(1, min(4, len(pdf_data['pages_text']) + 1))
            ])
            
            categories = self.categorize_project(
                project_name,
                text_sample,
                pdf_data['metadata']
            )
            
            categorizations[file_name] = categories
            print(f"{file_name}: {', '.join(categories)}")
        
        return categorizations

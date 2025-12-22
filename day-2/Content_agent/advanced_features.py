# Advanced Content Agent - Enhanced Features
# This is an extended version with memory, persistence, and advanced capabilities

import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# ============================================================================
# ADVANCED FEATURE 1: Persistent Memory System
# ============================================================================

class ContentMemory:
    """Store and retrieve generated content ideas for consistency."""
    
    def __init__(self, db_path: str = "content_agent.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing ideas."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ideas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                audience TEXT,
                title TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                performance_score REAL DEFAULT 0.0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS outlines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                idea_id INTEGER,
                content_type TEXT,
                outline_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (idea_id) REFERENCES ideas(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand_voice TEXT,
                niche TEXT,
                values TEXT,
                expertise TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_idea(self, topic: str, audience: str, title: str, 
                  description: str) -> int:
        """Save a content idea to memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ideas (topic, audience, title, description)
            VALUES (?, ?, ?, ?)
        ''', (topic, audience, title, description))
        
        conn.commit()
        idea_id = cursor.lastrowid
        conn.close()
        
        return idea_id
    
    def get_ideas(self, topic: str = None, limit: int = 10) -> List[Dict]:
        """Retrieve saved ideas."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if topic:
            cursor.execute('''
                SELECT * FROM ideas WHERE topic LIKE ? 
                ORDER BY created_at DESC LIMIT ?
            ''', (f"%{topic}%", limit))
        else:
            cursor.execute('''
                SELECT * FROM ideas 
                ORDER BY created_at DESC LIMIT ?
            ''', (limit,))
        
        ideas = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return ideas
    
    def save_outline(self, idea_id: int, content_type: str, 
                    outline_text: str) -> int:
        """Save an outline for an idea."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO outlines (idea_id, content_type, outline_text)
            VALUES (?, ?, ?)
        ''', (idea_id, content_type, outline_text))
        
        conn.commit()
        outline_id = cursor.lastrowid
        conn.close()
        
        return outline_id
    
    def set_preferences(self, brand_voice: str, niche: str, 
                       values: str, expertise: str):
        """Store user preferences for personalization."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_preferences (brand_voice, niche, values, expertise)
            VALUES (?, ?, ?, ?)
        ''', (brand_voice, niche, values, expertise))
        
        conn.commit()
        conn.close()
    
    def get_preferences(self) -> Dict:
        """Get most recent user preferences."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM user_preferences 
            ORDER BY created_at DESC LIMIT 1
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'brand_voice': row[1],
                'niche': row[2],
                'values': row[3],
                'expertise': row[4]
            }
        return None


# ============================================================================
# ADVANCED FEATURE 2: Content Evaluation System
# ============================================================================

class ContentEvaluator:
    """Evaluate quality of generated content ideas."""
    
    @staticmethod
    def score_idea(idea: str, audience: str, niche: str) -> float:
        """
        Score an idea on multiple dimensions:
        - Originality (0-25)
        - Relevance (0-25)
        - Clarity (0-25)
        - Actionability (0-25)
        """
        score = 0
        
        # Check originality (simpler ideas, common words = lower score)
        common_words = ['tips', 'guide', 'how to', 'introduction']
        originality = 20
        for word in common_words:
            if word.lower() in idea.lower():
                originality -= 3
        score += max(5, originality)
        
        # Check relevance to audience and niche
        relevance = 20
        if niche.lower() in idea.lower() or audience.lower() in idea.lower():
            relevance += 5
        score += relevance
        
        # Check clarity and specificity
        clarity = 20
        if len(idea) > 30:  # Specific ideas tend to be longer
            clarity += 5
        score += clarity
        
        # Check actionability
        action_words = ['learn', 'build', 'create', 'master', 'develop']
        actionability = 15
        for word in action_words:
            if word.lower() in idea.lower():
                actionability += 5
                break
        score += min(25, actionability)
        
        return min(100, score)
    
    @staticmethod
    def rank_ideas(ideas: List[str], audience: str, niche: str) -> List[tuple]:
        """Rank ideas by quality score."""
        scored_ideas = [
            (idea, ContentEvaluator.score_idea(idea, audience, niche))
            for idea in ideas
        ]
        
        # Sort by score descending
        scored_ideas.sort(key=lambda x: x[1], reverse=True)
        
        return scored_ideas


# ============================================================================
# ADVANCED FEATURE 3: Content Calendar Generator
# ============================================================================

class ContentCalendar:
    """Generate publishing calendars from content ideas."""
    
    @staticmethod
    def generate_calendar(ideas: List[str], time_frame: str = "4 weeks",
                         platforms: List[str] = None) -> Dict:
        """
        Generate a content calendar with scheduling.
        
        platforms: ['blog', 'twitter', 'youtube', 'linkedin', 'instagram']
        """
        if platforms is None:
            platforms = ['blog', 'twitter', 'linkedin']
        
        from datetime import datetime, timedelta
        
        calendar = {
            "generated_at": datetime.now().isoformat(),
            "time_frame": time_frame,
            "platforms": platforms,
            "schedule": []
        }
        
        # Parse time frame
        if "week" in time_frame:
            num_weeks = int(time_frame.split()[0])
            days = num_weeks * 7
        elif "month" in time_frame:
            num_months = int(time_frame.split()[0])
            days = num_months * 30
        else:
            days = 28
        
        start_date = datetime.now()
        ideas_per_day = max(1, len(ideas) / (days // 7))
        
        # Distribute ideas across platforms
        current_date = start_date
        idea_index = 0
        
        while (current_date - start_date).days < days and idea_index < len(ideas):
            for platform in platforms:
                if idea_index < len(ideas):
                    entry = {
                        "date": current_date.strftime("%Y-%m-%d"),
                        "day": current_date.strftime("%A"),
                        "platform": platform,
                        "idea": ideas[idea_index],
                        "optimal_time": ContentCalendar.get_optimal_time(platform),
                        "format": ContentCalendar.get_format_suggestion(platform)
                    }
                    calendar["schedule"].append(entry)
                    idea_index += 1
            
            current_date += timedelta(days=1)
        
        return calendar
    
    @staticmethod
    def get_optimal_time(platform: str) -> str:
        """Get optimal posting time for platform."""
        times = {
            "blog": "09:00 AM",
            "twitter": "08:00 AM, 12:00 PM, 06:00 PM",
            "linkedin": "07:30 AM, 12:00 PM, 05:30 PM",
            "youtube": "04:00 PM",
            "instagram": "11:00 AM, 07:00 PM"
        }
        return times.get(platform, "10:00 AM")
    
    @staticmethod
    def get_format_suggestion(platform: str) -> str:
        """Get format recommendation for platform."""
        formats = {
            "blog": "Long-form article (1000-3000 words)",
            "twitter": "Thread (5-10 tweets) or snippet",
            "linkedin": "Article or professional insight",
            "youtube": "5-15 minute video",
            "instagram": "Carousel (5-10 slides) or Reel"
        }
        return formats.get(platform, "Adapt to platform")


# ============================================================================
# ADVANCED FEATURE 4: Analytics & Performance Tracking
# ============================================================================

class ContentAnalytics:
    """Track and analyze content performance."""
    
    def __init__(self, db_path: str = "content_analytics.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize analytics database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                idea_id INTEGER,
                views INTEGER DEFAULT 0,
                engagement_rate REAL DEFAULT 0.0,
                shares INTEGER DEFAULT 0,
                comments INTEGER DEFAULT 0,
                conversion_rate REAL DEFAULT 0.0,
                tracked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_performance(self, idea_id: int, views: int = 0,
                       engagement_rate: float = 0.0, shares: int = 0,
                       comments: int = 0, conversion_rate: float = 0.0):
        """Log performance metrics for content."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_metrics 
            (idea_id, views, engagement_rate, shares, comments, conversion_rate)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (idea_id, views, engagement_rate, shares, comments, conversion_rate))
        
        conn.commit()
        conn.close()
    
    def get_top_performers(self, limit: int = 10) -> List[Dict]:
        """Get top performing content ideas."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT idea_id, SUM(views) as total_views, 
                   AVG(engagement_rate) as avg_engagement
            FROM performance_metrics
            GROUP BY idea_id
            ORDER BY total_views DESC
            LIMIT ?
        ''', (limit,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Initialize advanced features
    memory = ContentMemory()
    evaluator = ContentEvaluator()
    analytics = ContentAnalytics()
    
    print("\n" + "="*70)
    print("Advanced Content Agent - Features Demo")
    print("="*70)
    
    # Example 1: Save ideas to memory
    print("\n1️⃣ Saving ideas to memory...")
    ideas = [
        "5 Python Tricks That Will Change How You Code",
        "Building Your First AI Agent in 30 Minutes",
        "Machine Learning for Complete Beginners"
    ]
    
    for idea in ideas:
        idea_id = memory.save_idea(
            topic="Machine Learning",
            audience="Beginners",
            title=idea,
            description=f"A comprehensive guide on {idea}"
        )
        print(f"   ✓ Saved: {idea} (ID: {idea_id})")
    
    # Example 2: Score and rank ideas
    print("\n2️⃣ Evaluating ideas...")
    ranked_ideas = evaluator.rank_ideas(ideas, "Beginners", "Machine Learning")
    for idea, score in ranked_ideas:
        print(f"   Score: {score:.1f}/100 - {idea}")
    
    # Example 3: Generate content calendar
    print("\n3️⃣ Generating content calendar...")
    calendar = ContentCalendar.generate_calendar(
        ideas,
        time_frame="2 weeks",
        platforms=['blog', 'twitter', 'linkedin']
    )
    print(f"   Calendar generated with {len(calendar['schedule'])} entries")
    print(f"   Time frame: {calendar['time_frame']}")
    print(f"   Platforms: {', '.join(calendar['platforms'])}")
    
    # Show first entry
    if calendar['schedule']:
        entry = calendar['schedule'][0]
        print(f"\n   First entry:")
        print(f"   - Date: {entry['date']} ({entry['day']})")
        print(f"   - Platform: {entry['platform']}")
        print(f"   - Optimal time: {entry['optimal_time']}")
        print(f"   - Format: {entry['format']}")
    
    # Example 4: Set preferences
    print("\n4️⃣ Storing user preferences...")
    memory.set_preferences(
        brand_voice="Professional but conversational",
        niche="AI & Machine Learning",
        values="Accessibility, Practical Learning",
        expertise="Python, Data Science, ML"
    )
    prefs = memory.get_preferences()
    print(f"   ✓ Preferences saved:")
    for key, value in prefs.items():
        print(f"     - {key.replace('_', ' ').title()}: {value}")
    
    # Example 5: Log performance
    print("\n5️⃣ Tracking content performance...")
    analytics.log_performance(
        idea_id=1,
        views=2500,
        engagement_rate=0.08,
        shares=45,
        comments=120
    )
    print("   ✓ Performance metrics logged")
    
    print("\n" + "="*70)
    print("✅ Advanced features initialized successfully!")
    print("="*70)
    
    # Save calendar to JSON
    calendar_path = "content_calendar.json"
    with open(calendar_path, 'w') as f:
        json.dump(calendar, f, indent=2)
    print(f"\n📅 Calendar saved to: {calendar_path}")

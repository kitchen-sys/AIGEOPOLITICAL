"""
GeoBot Discord Bot

Real-time geopolitical intelligence bot for Discord with:
- Auto-posting ticker updates every 5 minutes
- /compare - Compare nations in conflict
- /scan - Scan conflict for escalation/regime change probabilities
- /ask - Answer geopolitical questions with GeoBot 2.0 analytics
"""

import asyncio
import os
from typing import Optional
from datetime import datetime

try:
    import discord
    from discord.ext import commands, tasks
    HAS_DISCORD = True
except ImportError:
    HAS_DISCORD = False
    discord = None
    commands = None
    tasks = None

try:
    from ..data_ingestion.rss_scraper import RSSFeedScraper
    HAS_RSS = True
except ImportError:
    HAS_RSS = False

try:
    from ..analysis.engine import AnalyticalEngine
    HAS_ANALYSIS = True
except ImportError:
    HAS_ANALYSIS = False

try:
    from .forecaster import ConflictForecaster
    HAS_FORECASTER = True
except ImportError:
    HAS_FORECASTER = False


class GeoBotDiscord(commands.Bot):
    """
    GeoBot Discord Bot for real-time geopolitical intelligence.
    """

    def __init__(self, ticker_channel_id: Optional[int] = None):
        """
        Initialize Discord bot.

        Parameters
        ----------
        ticker_channel_id : Optional[int]
            Discord channel ID for auto-posting ticker updates
        """
        if not HAS_DISCORD:
            raise ImportError("discord.py is required. Install with: pip install discord.py")

        # Initialize bot with intents
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='/', intents=intents)

        self.ticker_channel_id = ticker_channel_id
        self.scraper = RSSFeedScraper() if HAS_RSS else None
        self.engine = AnalyticalEngine() if HAS_ANALYSIS else None
        self.forecaster = ConflictForecaster() if HAS_FORECASTER else None

        # Track seen articles for ticker
        self.seen_articles = set()

        # Register commands
        self.add_commands()

    def add_commands(self):
        """Register all bot commands."""

        @self.command(name='scan', help='Scan conflict for escalation and regime change probabilities')
        async def scan(ctx, *, conflict: str):
            """
            Scan a conflict for escalation and regime change probabilities.

            Usage: /scan <conflict name>
            Example: /scan taiwan
            """
            if not HAS_FORECASTER:
                await ctx.send("⚠️ Forecasting module not available")
                return

            await ctx.send(f"🔍 Analyzing conflict: **{conflict}**...")

            try:
                # Get recent news for context
                context_articles = []
                if self.scraper:
                    articles = self.scraper.scrape_all(geopolitical_only=True)
                    # Filter articles relevant to this conflict
                    conflict_lower = conflict.lower()
                    relevant = [a for a in articles if conflict_lower in (a.title + a.summary).lower()]
                    context_articles = [a.title + " " + a.summary for a in relevant[:5]]

                # Generate forecast
                forecast = self.forecaster.forecast_conflict(
                    conflict,
                    context_text=" ".join(context_articles) if context_articles else None
                )

                # Create embed
                embed = discord.Embed(
                    title=f"⚔️ Conflict Analysis: {forecast.conflict_name}",
                    description=f"**Risk Level: {forecast.risk_level.upper()}**",
                    color=self._get_risk_color(forecast.risk_level),
                    timestamp=datetime.utcnow()
                )

                embed.add_field(
                    name="📈 Escalation Probability",
                    value=f"**{forecast.escalation_probability:.1%}**",
                    inline=True
                )

                embed.add_field(
                    name="🏛️ Regime Change Probability",
                    value=f"**{forecast.regime_change_probability:.1%}**",
                    inline=True
                )

                embed.add_field(
                    name="⏰ Timeframe",
                    value=forecast.timeframe,
                    inline=True
                )

                embed.add_field(
                    name="🎯 Confidence",
                    value=f"{forecast.confidence:.0%}",
                    inline=True
                )

                embed.add_field(
                    name="📊 Key Factors",
                    value="\n".join([f"• {factor}" for factor in forecast.key_factors]),
                    inline=False
                )

                embed.set_footer(text="GeoBot 2.0 | Powered by Bayesian Forecasting")

                await ctx.send(embed=embed)

            except Exception as e:
                await ctx.send(f"❌ Error analyzing conflict: {str(e)}")

        @self.command(name='compare', help='Compare two nations in conflict context')
        async def compare(ctx, nation1: str, nation2: str):
            """
            Compare two nations in conflict context.

            Usage: /compare <nation1> <nation2>
            Example: /compare China Taiwan
            """
            if not HAS_FORECASTER:
                await ctx.send("⚠️ Comparison module not available")
                return

            await ctx.send(f"🔄 Comparing **{nation1}** vs **{nation2}**...")

            try:
                # Get comparison analysis
                comparison = self.forecaster.compare_nations(nation1, nation2)

                # Create embed
                embed = discord.Embed(
                    title=f"⚖️ Nation Comparison: {nation1} vs {nation2}",
                    color=discord.Color.blue(),
                    timestamp=datetime.utcnow()
                )

                # Add analysis
                if len(comparison['analysis']) > 1024:
                    # Split if too long
                    parts = [comparison['analysis'][i:i+1024] for i in range(0, len(comparison['analysis']), 1024)]
                    for i, part in enumerate(parts[:3], 1):  # Max 3 parts
                        embed.add_field(
                            name=f"Analysis (Part {i})" if i > 1 else "Analysis",
                            value=part,
                            inline=False
                        )
                else:
                    embed.add_field(
                        name="Analysis",
                        value=comparison['analysis'],
                        inline=False
                    )

                embed.set_footer(text=f"GeoBot 2.0 | Method: {comparison['method']}")

                await ctx.send(embed=embed)

            except Exception as e:
                await ctx.send(f"❌ Error comparing nations: {str(e)}")

        @self.command(name='ask', help='Ask any geopolitical question')
        async def ask(ctx, *, question: str):
            """
            Ask any geopolitical question and get analytics-powered answer.

            Usage: /ask <question>
            Example: /ask What are the logistics challenges in Taiwan strait?
            """
            if not HAS_ANALYSIS:
                await ctx.send("⚠️ Analytical engine not available. Install GeoBot 2.0 modules.")
                return

            await ctx.send(f"🤔 Analyzing: *{question}*...")

            try:
                # Build context from recent news
                context = {'question': question}

                if self.scraper:
                    articles = self.scraper.scrape_all(geopolitical_only=True)
                    context['recent_news'] = [
                        {'title': a.title, 'source': a.source}
                        for a in articles[:10]
                    ]

                # Generate analysis
                analysis = self.engine.analyze(question, context)

                # Create embed
                embed = discord.Embed(
                    title="💡 GeoBot 2.0 Analysis",
                    description=f"**Question:** {question}",
                    color=discord.Color.green(),
                    timestamp=datetime.utcnow()
                )

                # Split analysis if too long
                if len(analysis) > 2048:
                    parts = [analysis[i:i+1024] for i in range(0, len(analysis), 1024)]
                    for i, part in enumerate(parts[:4], 1):  # Max 4 fields
                        embed.add_field(
                            name=f"Response (Part {i})" if i > 1 else "Response",
                            value=part,
                            inline=False
                        )
                else:
                    embed.add_field(
                        name="Response",
                        value=analysis,
                        inline=False
                    )

                embed.set_footer(text="GeoBot 2.0 | Clinical Systems Analysis")

                await ctx.send(embed=embed)

            except Exception as e:
                await ctx.send(f"❌ Error processing question: {str(e)}")

        @self.command(name='status', help='Check bot status and module availability')
        async def status(ctx):
            """Show bot status and available modules."""
            embed = discord.Embed(
                title="🤖 GeoBot Status",
                color=discord.Color.purple(),
                timestamp=datetime.utcnow()
            )

            embed.add_field(
                name="Modules",
                value=f"RSS Scraper: {'✅' if HAS_RSS else '❌'}\n"
                      f"GeoBot 2.0: {'✅' if HAS_ANALYSIS else '❌'}\n"
                      f"Forecasting: {'✅' if HAS_FORECASTER else '❌'}",
                inline=True
            )

            embed.add_field(
                name="Ticker",
                value=f"Channel: {self.ticker_channel_id or 'Not configured'}\n"
                      f"Status: {'Running' if self.ticker_loop.is_running() else 'Stopped'}",
                inline=True
            )

            embed.add_field(
                name="Commands",
                value="/scan - Analyze conflicts\n"
                      "/compare - Compare nations\n"
                      "/ask - Ask questions\n"
                      "/status - This message",
                inline=False
            )

            embed.set_footer(text="GeoBot 2.0 Discord Bot")

            await ctx.send(embed=embed)

    def _get_risk_color(self, risk_level: str) -> discord.Color:
        """Get Discord color for risk level."""
        colors = {
            'critical': discord.Color.dark_red(),
            'high': discord.Color.red(),
            'medium': discord.Color.orange(),
            'low': discord.Color.green()
        }
        return colors.get(risk_level, discord.Color.blue())

    async def on_ready(self):
        """Called when bot is ready."""
        print(f'✅ GeoBot logged in as {self.user}')
        print(f'📡 Connected to {len(self.guilds)} server(s)')

        if self.ticker_channel_id:
            print(f'📰 Ticker channel: {self.ticker_channel_id}')
            if not self.ticker_loop.is_running():
                self.ticker_loop.start()
        else:
            print('⚠️ No ticker channel configured')

    @tasks.loop(minutes=5)
    async def ticker_loop(self):
        """Post ticker updates every 5 minutes."""
        if not self.ticker_channel_id or not self.scraper:
            return

        try:
            channel = self.get_channel(self.ticker_channel_id)
            if not channel:
                print(f"⚠️ Could not find ticker channel {self.ticker_channel_id}")
                return

            # Scrape news
            articles = self.scraper.scrape_all(geopolitical_only=True)

            # Filter new articles
            new_articles = []
            for article in articles:
                if article.link not in self.seen_articles:
                    new_articles.append(article)
                    self.seen_articles.add(article.link)

            if not new_articles:
                return  # No new articles, skip update

            # Create ticker update
            embed = discord.Embed(
                title="📰 Geopolitical Intelligence Update",
                description=f"**{len(new_articles)} new developments**",
                color=discord.Color.blue(),
                timestamp=datetime.utcnow()
            )

            # Add top 5 articles
            for i, article in enumerate(new_articles[:5], 1):
                countries = article.extract_countries()
                countries_str = ", ".join(countries[:3]) if countries else "Global"

                embed.add_field(
                    name=f"{i}. {article.title[:100]}",
                    value=f"🌍 {countries_str} | 📰 {article.source}\n[Read more]({article.link})",
                    inline=False
                )

            if len(new_articles) > 5:
                embed.add_field(
                    name="Additional Updates",
                    value=f"...and {len(new_articles) - 5} more developments",
                    inline=False
                )

            embed.set_footer(text="GeoBot 2.0 | Auto-update every 5 minutes")

            await channel.send(embed=embed)

        except Exception as e:
            print(f"❌ Error in ticker loop: {e}")

    @ticker_loop.before_loop
    async def before_ticker(self):
        """Wait until bot is ready before starting ticker."""
        await self.wait_until_ready()


def run_discord_bot(token: str, ticker_channel_id: Optional[int] = None):
    """
    Run the GeoBot Discord bot.

    Parameters
    ----------
    token : str
        Discord bot token
    ticker_channel_id : Optional[int]
        Channel ID for auto-posting ticker updates
    """
    if not HAS_DISCORD:
        raise ImportError(
            "discord.py is required for Discord bot. "
            "Install with: pip install discord.py"
        )

    bot = GeoBotDiscord(ticker_channel_id=ticker_channel_id)

    print("=" * 80)
    print("GEOBOT DISCORD BOT")
    print("=" * 80)
    print()
    print("Commands:")
    print("  /scan <conflict> - Analyze conflict escalation")
    print("  /compare <nation1> <nation2> - Compare nations")
    print("  /ask <question> - Ask geopolitical questions")
    print("  /status - Check bot status")
    print()
    print("Starting bot...")
    print("=" * 80)

    bot.run(token)


if __name__ == '__main__':
    # Run bot with token from environment
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        print("Error: DISCORD_BOT_TOKEN environment variable not set")
        print("Set it with: export DISCORD_BOT_TOKEN='your_token_here'")
        exit(1)

    channel_id = os.getenv('TICKER_CHANNEL_ID')
    if channel_id:
        channel_id = int(channel_id)

    run_discord_bot(token, channel_id)

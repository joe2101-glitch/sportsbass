#!/usr/bin/env python3
"""
SportsBass AI — Daily Data Pipeline
====================================
Fetches live data from free APIs, recalculates model ratings,
and outputs JSON files the app can load.

APIs used:
  - NHL: api-web.nhle.com (free, no key)
  - EPL: football-data.org (free tier, needs API key — 10 req/min)

Schedule: Run daily via GitHub Actions (see .github/workflows/update.yml)
Output:   epl_data.json, nhl_data.json → push to GitHub Pages / CDN

Usage:
  pip install requests numpy scipy
  python update_data.py --epl-key YOUR_FOOTBALL_DATA_API_KEY
"""

import json, math, argparse, sys
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import requests
    import numpy as np
    from scipy.optimize import minimize
except ImportError:
    print("Install deps: pip install requests numpy scipy")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# EPL PIPELINE
# ═══════════════════════════════════════════════════════════════

# Team colors (static — won't change)
EPL_COLORS = {
    "Arsenal FC": ("#EF0107","#fff"), "Aston Villa FC": ("#670E36","#95BFE5"),
    "AFC Bournemouth": ("#DA291C","#000"), "Brentford FC": ("#e30613","#FFB81C"),
    "Brighton & Hove Albion FC": ("#0057B8","#fff"), "Burnley FC": ("#6C1D45","#99D6EA"),
    "Chelsea FC": ("#034694","#fff"), "Crystal Palace FC": ("#1B458F","#C4122E"),
    "Everton FC": ("#003399","#fff"), "Fulham FC": ("#000","#fff"),
    "Leeds United FC": ("#FFCD00","#1D428A"), "Liverpool FC": ("#C8102E","#fff"),
    "Manchester City FC": ("#6CABDD","#fff"), "Manchester United FC": ("#DA291C","#FBE122"),
    "Newcastle United FC": ("#241F20","#fff"), "Nottingham Forest FC": ("#DD0000","#fff"),
    "Sheffield United FC": ("#EE2737","#000"), "Southampton FC": ("#D71920","#fff"),
    "Tottenham Hotspur FC": ("#132257","#fff"), "West Ham United FC": ("#7A263A","#1BB1E7"),
    "Wolverhampton Wanderers FC": ("#FDB913","#000"), "Leicester City FC": ("#003090","#FDBE11"),
    "Ipswich Town FC": ("#0033A0","#fff"), "Sunderland AFC": ("#EB172B","#fff"),
}

EPL_SHORT_NAMES = {
    "Arsenal FC": "Arsenal", "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth", "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton", "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea", "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton", "Fulham FC": "Fulham",
    "Leeds United FC": "Leeds", "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City", "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle", "Nottingham Forest FC": "Nott'm Forest",
    "Sheffield United FC": "Sheffield Utd", "Southampton FC": "Southampton",
    "Tottenham Hotspur FC": "Tottenham", "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves", "Leicester City FC": "Leicester",
    "Ipswich Town FC": "Ipswich", "Sunderland AFC": "Sunderland",
}


def fetch_epl_data(api_key, season="2025"):
    """Fetch EPL matches from football-data.org"""
    headers = {"X-Auth-Token": api_key}
    base = "https://api.football-data.org/v4"
    
    # Get all matches for current season
    url = f"{base}/competitions/PL/matches?season={season}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    
    matches = []
    for m in data.get("matches", []):
        if m["status"] != "FINISHED":
            continue
        matches.append({
            "home": m["homeTeam"]["name"],
            "away": m["awayTeam"]["name"],
            "hg": m["score"]["fullTime"]["home"],
            "ag": m["score"]["fullTime"]["away"],
            "date": m["utcDate"][:10],
        })
    
    # Get upcoming fixtures
    fixtures = []
    for m in data.get("matches", []):
        if m["status"] in ("SCHEDULED", "TIMED"):
            match_date = m["utcDate"][:10]
            today = datetime.utcnow().strftime("%Y-%m-%d")
            if match_date == today:
                kick = m["utcDate"][11:16]  # HH:MM UTC
                fixtures.append({
                    "home": EPL_SHORT_NAMES.get(m["homeTeam"]["name"], m["homeTeam"]["name"]),
                    "away": EPL_SHORT_NAMES.get(m["awayTeam"]["name"], m["awayTeam"]["name"]),
                    "time": kick,
                })
    
    return matches, fixtures


def fit_dixon_coles(matches, half_life_days=120):
    """Fit Dixon-Coles model with time-weighted MLE"""
    teams = sorted(set(m["home"] for m in matches) | set(m["away"] for m in matches))
    team_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    
    # Time weights (exponential decay)
    today = datetime.utcnow()
    weights = []
    for m in matches:
        d = (today - datetime.strptime(m["date"], "%Y-%m-%d")).days
        weights.append(math.exp(-d / half_life_days))
    
    def neg_log_lik(params):
        att = params[:n]
        defe = params[n:2*n]
        home_adv = params[2*n]
        rho = params[2*n+1]
        
        ll = 0
        for k, m in enumerate(matches):
            hi, ai = team_idx[m["home"]], team_idx[m["away"]]
            lam_h = math.exp(home_adv + att[hi] + defe[ai])
            lam_a = math.exp(att[ai] + defe[hi])
            
            hg, ag = m["hg"], m["ag"]
            
            # Poisson PMF
            p_h = math.exp(-lam_h) * (lam_h ** hg) / math.factorial(hg)
            p_a = math.exp(-lam_a) * (lam_a ** ag) / math.factorial(ag)
            
            # Dixon-Coles correction
            if hg == 0 and ag == 0:
                tau = 1 - lam_h * lam_a * rho
            elif hg == 1 and ag == 0:
                tau = 1 + lam_a * rho
            elif hg == 0 and ag == 1:
                tau = 1 + lam_h * rho
            elif hg == 1 and ag == 1:
                tau = 1 - rho
            else:
                tau = 1
            
            p = p_h * p_a * max(tau, 1e-10)
            ll += weights[k] * math.log(max(p, 1e-15))
        
        # Constraint: sum of attack = 0
        ll -= 100 * (sum(att) ** 2)
        return -ll
    
    # Initial params
    x0 = np.zeros(2*n + 2)
    x0[2*n] = 0.2   # home advantage
    x0[2*n+1] = -0.1  # rho
    
    result = minimize(neg_log_lik, x0, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-10})
    
    params = result.x
    att = {teams[i]: round(float(params[i]), 4) for i in range(n)}
    defe = {teams[i]: round(float(params[n+i]), 4) for i in range(n)}
    home_adv = round(float(params[2*n]), 4)
    rho = round(float(params[2*n+1]), 4)
    
    return att, defe, home_adv, rho, teams


def calc_epl_stats(matches, teams):
    """Calculate per-team stats from match data"""
    stats = defaultdict(lambda: {"gf": 0, "ga": 0, "gp": 0, "w": 0, "d": 0, "l": 0, "results": []})
    
    for m in matches:
        h, a = m["home"], m["away"]
        hg, ag = m["hg"], m["ag"]
        
        for t, gf, ga, is_home in [(h, hg, ag, True), (a, ag, hg, False)]:
            stats[t]["gf"] += gf
            stats[t]["ga"] += ga
            stats[t]["gp"] += 1
            if gf > ga:
                stats[t]["w"] += 1; stats[t]["results"].append(("W", m["date"]))
            elif gf < ga:
                stats[t]["l"] += 1; stats[t]["results"].append(("L", m["date"]))
            else:
                stats[t]["d"] += 1; stats[t]["results"].append(("D", m["date"]))
    
    out = {}
    for t in teams:
        s = stats[t]
        gp = max(s["gp"], 1)
        ppg = round((s["w"] * 3 + s["d"]) / gp, 2)
        gf_pg = round(s["gf"] / gp, 2)
        ga_pg = round(s["ga"] / gp, 2)
        # Last 6 form
        recent = sorted(s["results"], key=lambda x: x[1])[-6:]
        form = "".join(r[0] for r in recent)
        
        out[t] = {"ppg": ppg, "gf": gf_pg, "ga": ga_pg, "form": form, "gp": gp}
    
    return out


def build_epl_h2h(matches, teams_short):
    """Build H2H lookup from match data"""
    h2h = defaultdict(list)
    
    for m in matches:
        h_short = teams_short.get(m["home"], m["home"])
        a_short = teams_short.get(m["away"], m["away"])
        h2h_key = "|".join(sorted([h_short, a_short]))
        h2h[h2h_key].append([h_short, m["hg"], m["ag"], a_short])
    
    # Keep last 5 per pair
    return {k: v[-5:] for k, v in h2h.items()}


def build_epl_json(api_key):
    """Full EPL pipeline"""
    print("Fetching EPL data...")
    matches, fixtures = fetch_epl_data(api_key)
    print(f"  {len(matches)} finished matches, {len(fixtures)} today's fixtures")
    
    print("Fitting Dixon-Coles model...")
    att, defe, home_adv, rho, teams = fit_dixon_coles(matches)
    
    print("Calculating stats...")
    team_stats = calc_epl_stats(matches, teams)
    
    print("Building H2H...")
    h2h = build_epl_h2h(matches, EPL_SHORT_NAMES)
    
    # Assemble output matching app format
    epl_teams = {}
    for t in teams:
        short = EPL_SHORT_NAMES.get(t, t)
        color, accent = EPL_COLORS.get(t, ("#666", "#fff"))
        s = team_stats[t]
        epl_teams[short] = {
            "att": att[t],
            "def": defe[t],
            "form": s["form"],
            "ppg": s["ppg"],
            "gf": s["gf"],
            "ga": s["ga"],
            "xgf": None,  # Would need understat/fbref scrape for xG
            "xga": None,
            "color": color,
            "accent": accent,
        }
    
    output = {
        "updated": datetime.utcnow().isoformat()[:19] + "Z",
        "season": "2025-26",
        "homeAdv": home_adv,
        "rho": rho,
        "teams": epl_teams,
        "h2h": h2h,
        "fixtures": fixtures,
    }
    
    with open("epl_data.json", "w") as f:
        json.dump(output, f, separators=(",", ":"))
    
    print(f"  Written epl_data.json ({len(epl_teams)} teams)")
    return output


# ═══════════════════════════════════════════════════════════════
# NHL PIPELINE
# ═══════════════════════════════════════════════════════════════

NHL_COLORS = {
    "ANA": "#F47A38", "BOS": "#FFB81C", "BUF": "#003087", "CAR": "#CC0000",
    "CBJ": "#002654", "CGY": "#D2001C", "CHI": "#CF0A2C", "COL": "#6F263D",
    "DAL": "#006847", "DET": "#CE1126", "EDM": "#041E42", "FLA": "#041E42",
    "LAK": "#A2AAAD", "MIN": "#154734", "MTL": "#AF1E2D", "NJD": "#CE1126",
    "NSH": "#FFB81C", "NYI": "#00539B", "NYR": "#0038A8", "OTT": "#C52032",
    "PHI": "#F74902", "PIT": "#FCB514", "SEA": "#99D9D9", "SJS": "#006D75",
    "STL": "#002F87", "TBL": "#002868", "TOR": "#00205B", "UTA": "#6CACE4",
    "VAN": "#00205B", "VGK": "#B4975A", "WPG": "#041E42", "WSH": "#C8102E",
}

NHL_DIVISIONS = {
    "ANA": "Pacific", "CGY": "Pacific", "EDM": "Pacific", "LAK": "Pacific",
    "SJS": "Pacific", "SEA": "Pacific", "VAN": "Pacific", "VGK": "Pacific",
    "CHI": "Central", "COL": "Central", "DAL": "Central", "MIN": "Central",
    "NSH": "Central", "STL": "Central", "UTA": "Central", "WPG": "Central",
    "BOS": "Atlantic", "BUF": "Atlantic", "DET": "Atlantic", "FLA": "Atlantic",
    "MTL": "Atlantic", "OTT": "Atlantic", "TBL": "Atlantic", "TOR": "Atlantic",
    "CAR": "Metropolitan", "CBJ": "Metropolitan", "NJD": "Metropolitan",
    "NYI": "Metropolitan", "NYR": "Metropolitan", "PHI": "Metropolitan",
    "PIT": "Metropolitan", "WSH": "Metropolitan",
}


def fetch_nhl_standings():
    """Fetch current NHL standings"""
    url = "https://api-web.nhle.com/v1/standings/now"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()["standings"]


def fetch_nhl_team_stats():
    """Fetch team stats from NHL API"""
    url = "https://api-web.nhle.com/v1/standings/now"
    resp = requests.get(url)
    resp.raise_for_status()
    standings = resp.json()["standings"]
    
    teams = {}
    league_gf = 0
    league_ga = 0
    league_gp = 0
    
    for t in standings:
        abbrev = t["teamAbbrev"]["default"]
        gp = t["gamesPlayed"]
        gf = t["goalFor"]
        ga = t["goalAgainst"]
        
        league_gf += gf
        league_ga += ga
        league_gp += gp
        
        teams[abbrev] = {
            "name": t["teamName"]["default"],
            "abbrev": abbrev,
            "gp": gp,
            "gf": gf, "ga": ga,
            "w": t["wins"], "l": t["losses"], "otl": t["otLosses"],
            "pts": t["points"],
            "homeRecord": t.get("homeWins", 0),
            "homeLosses": t.get("homeLosses", 0),
            "homeOtl": t.get("homeOtLosses", 0),
            "awayRecord": t.get("roadWins", 0),
            "awayLosses": t.get("roadLosses", 0),
            "awayOtl": t.get("roadOtLosses", 0),
            "streakCode": t.get("streakCode", ""),
            "streakCount": t.get("streakCount", 0),
        }
    
    league_avg = league_gf / max(league_gp, 1) * 2  # goals per game (both teams)
    
    return teams, league_avg / 2  # per-team avg


def fetch_nhl_schedule_today():
    """Fetch today's NHL games"""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    url = f"https://api-web.nhle.com/v1/schedule/{today}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    
    fixtures = []
    for day in data.get("gameWeek", []):
        if day["date"] == today:
            for game in day.get("games", []):
                fixtures.append({
                    "home": game["homeTeam"]["abbrev"],
                    "away": game["awayTeam"]["abbrev"],
                    "time": game.get("startTimeUTC", "")[:16].split("T")[-1] if "startTimeUTC" in game else "TBD",
                })
    
    return fixtures


def fetch_nhl_recent_results():
    """Fetch recent game results for form calculation"""
    end = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    url = f"https://api-web.nhle.com/v1/schedule/{start}"
    # NHL API returns a week at a time from the given date
    # We'll fetch multiple weeks
    
    results = defaultdict(list)
    current = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    
    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        try:
            resp = requests.get(f"https://api-web.nhle.com/v1/schedule/{date_str}")
            if resp.status_code == 200:
                data = resp.json()
                for day in data.get("gameWeek", []):
                    for game in day.get("games", []):
                        if game.get("gameState") == "OFF":
                            home = game["homeTeam"]["abbrev"]
                            away = game["awayTeam"]["abbrev"]
                            hs = game["homeTeam"].get("score", 0)
                            as_ = game["awayTeam"].get("score", 0)
                            date = day["date"]
                            
                            if hs > as_:
                                results[home].append(("W", date))
                                # Check if OT
                                period = game.get("periodDescriptor", {}).get("number", 3)
                                results[away].append(("OTL" if period > 3 else "L", date))
                            else:
                                results[away].append(("W", date))
                                period = game.get("periodDescriptor", {}).get("number", 3)
                                results[home].append(("OTL" if period > 3 else "L", date))
        except:
            pass
        current += timedelta(days=7)
    
    # Return last 6 for each team
    form = {}
    for team, res in results.items():
        recent = sorted(res, key=lambda x: x[1])[-6:]
        form[team] = [r[0] for r in recent]
    
    return form


def build_nhl_json():
    """Full NHL pipeline"""
    print("Fetching NHL standings...")
    teams_raw, league_avg = fetch_nhl_team_stats()
    
    print("Fetching today's schedule...")
    fixtures = fetch_nhl_schedule_today()
    
    print("Fetching recent results for form...")
    form_data = fetch_nhl_recent_results()
    
    # Calculate attack/defense ratings
    teams_out = {}
    for abbrev, t in teams_raw.items():
        gp = max(t["gp"], 1)
        gf_pg = round(t["gf"] / gp, 3)
        ga_pg = round(t["ga"] / gp, 3)
        attack = round(gf_pg / league_avg, 4)
        defense = round(ga_pg / league_avg, 4)
        
        home_rec = f"{t['homeRecord']}-{t['homeLosses']}-{t['homeOtl']}"
        away_rec = f"{t['awayRecord']}-{t['awayLosses']}-{t['awayOtl']}"
        
        form = form_data.get(abbrev, [])
        
        teams_out[abbrev] = {
            "name": t["name"],
            "abbrev": abbrev,
            "color": NHL_COLORS.get(abbrev, "#666"),
            "division": NHL_DIVISIONS.get(abbrev, ""),
            "attack": attack,
            "defense": defense,
            "gf_pg": gf_pg,
            "ga_pg": ga_pg,
            "gp": t["gp"],
            "GF": t["gf"], "GA": t["ga"],
            "w": t["w"], "l": t["l"], "otl": t["otl"],
            "pts": t["pts"],
            "home": home_rec,
            "away": away_rec,
            "form": form,
            # Placeholder stats — would need scraping for advanced
            "sogF": 28.0, "sogA": 28.0,
            "svPct": 89.0, "shPct": 11.0,
            "ppPct": 24.0, "pkPct": 76.0,
            "xgf_pg": gf_pg, "xga_pg": ga_pg,  # Approximation
            "xgPct": 0.50,
            "corsiPct": 0.50,
        }
    
    output = {
        "updated": datetime.utcnow().isoformat()[:19] + "Z",
        "season": "2025-26",
        "leagueAvg": {
            "sogF": 28.0, "sogA": 28.0, "svPct": 89.0, "shPct": 11.0,
            "ppPct": 24.4, "pkPct": 75.7, "foPct": 50.0,
        },
        "homeAdv": 0.12,
        "dcRho": -0.04,
        "xgWeight": 0.25,
        "teams": teams_out,
        "h2h": {},  # Would need game-by-game fetch to build
        "fixtures": fixtures,
    }
    
    with open("nhl_data.json", "w") as f:
        json.dump(output, f, separators=(",", ":"))
    
    print(f"  Written nhl_data.json ({len(teams_out)} teams, {len(fixtures)} fixtures today)")
    return output


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SportsBass AI Data Pipeline")
    parser.add_argument("--epl-key", help="football-data.org API key")
    parser.add_argument("--nhl-only", action="store_true", help="Only update NHL")
    parser.add_argument("--epl-only", action="store_true", help="Only update EPL")
    args = parser.parse_args()
    
    if not args.epl_only:
        build_nhl_json()
    
    if not args.nhl_only:
        if args.epl_key:
            build_epl_json(args.epl_key)
        else:
            print("Skipping EPL (no --epl-key provided)")
            print("  Get free key at: https://www.football-data.org/client/register")
    
    print("\n✅ Done! Upload epl_data.json and nhl_data.json to your CDN/GitHub Pages.")

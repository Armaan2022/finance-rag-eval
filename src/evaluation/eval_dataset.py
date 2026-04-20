EVAL_QUESTIONS = [
    # AAPL — Financials
    {
        "question": "What was Apple's total net sales in fiscal year 2025?",
        "ground_truth": "Apple's total net sales in fiscal year 2025 were $416,161 million.",
        "ticker": "AAPL",
        "category": "financials",
    },
    {
        "question": "What was Apple's net income in fiscal year 2025?",
        "ground_truth": "Apple's net income in fiscal year 2025 was $101,389 million.",
        "ticker": "AAPL",
        "category": "financials",
    },
    {
        "question": "How much did Apple spend on research and development in 2025?",
        "ground_truth": "Apple spent $29,915 million on research and development in fiscal year 2025.",
        "ticker": "AAPL",
        "category": "financials",
    },
    {
        "question": "What was Apple's quarterly cash dividend per share as of September 2025?",
        "ground_truth": "Apple's quarterly cash dividend was $0.26 per share as of September 27, 2025.",
        "ticker": "AAPL",
        "category": "financials",
    },
    {
        "question": "How many shares did Apple repurchase in fiscal year 2025?",
        "ground_truth": "Apple repurchased 402 million shares during fiscal year 2025.",
        "ticker": "AAPL",
        "category": "financials",
    },
    # AAPL — Business & Strategy
    {
        "question": "What are Apple's main product categories?",
        "ground_truth": "Apple's main product categories are iPhone, Mac, iPad, Wearables/Home/Accessories, and Services.",
        "ticker": "AAPL",
        "category": "strategy",
    },
    {
        "question": "What geographic segments does Apple report revenue for?",
        "ground_truth": "Apple reports revenue for the Americas, Europe, Greater China, Japan, and Rest of Asia Pacific.",
        "ticker": "AAPL",
        "category": "strategy",
    },
    {
        "question": "What is Apple's largest revenue segment by geography?",
        "ground_truth": "The Americas is Apple's largest geographic segment, generating $151,790 million in fiscal year 2025.",
        "ticker": "AAPL",
        "category": "strategy",
    },
    # AAPL — Risk Factors
    {
        "question": "What supply chain risks does Apple identify in its 10-K?",
        "ground_truth": "Apple identifies risks including reliance on single or limited sources for critical components, vulnerability to supply shortages and price increases, and risks from business interruptions at key suppliers.",
        "ticker": "AAPL",
        "category": "risk",
    },
    {
        "question": "What macroeconomic risks does Apple mention?",
        "ground_truth": "Apple mentions risks from global economic conditions, inflation, tariffs, trade policy changes, and geopolitical tensions that could adversely affect demand for its products and services.",
        "ticker": "AAPL",
        "category": "risk",
    },

    # MSFT — Financials
    {
        "question": "What was Microsoft's total revenue in fiscal year 2025?",
        "ground_truth": "Microsoft's total revenue in fiscal year 2025 was $279,600 million (approximately $279.6 billion).",
        "ticker": "MSFT",
        "category": "financials",
    },
    {
        "question": "How much did Microsoft spend on research and development in fiscal year 2025?",
        "ground_truth": "Microsoft spent approximately $33 billion on research and development in fiscal year 2025, a 10% increase from fiscal year 2024.",
        "ticker": "MSFT",
        "category": "financials",
    },
    {
        "question": "What was Microsoft's diluted earnings per share in fiscal year 2025?",
        "ground_truth": "Microsoft's diluted earnings per share in fiscal year 2025 was $13.64.",
        "ticker": "MSFT",
        "category": "financials",
    },

    # MSFT — Business & Strategy
    {
        "question": "What are Microsoft's three main business segments?",
        "ground_truth": "Microsoft's three main business segments are Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.",
        "ticker": "MSFT",
        "category": "strategy",
    },
    {
        "question": "What is Microsoft's relationship with OpenAI?",
        "ground_truth": "Microsoft is a major investor in OpenAI and the companies have a reciprocal revenue-sharing arrangement. Microsoft integrates OpenAI models into its products and provides Azure cloud infrastructure for OpenAI.",
        "ticker": "MSFT",
        "category": "strategy",
    },
    {
        "question": "What drove growth in Microsoft's Intelligent Cloud segment?",
        "ground_truth": "Intelligent Cloud revenue growth was primarily driven by Azure cloud services, including AI services and infrastructure demand.",
        "ticker": "MSFT",
        "category": "strategy",
    },

    # MSFT — Risk Factors
    {
        "question": "What AI-related risks does Microsoft identify?",
        "ground_truth": "Microsoft identifies risks including AI algorithms being flawed, datasets being overbroad or insufficient, potential for harmful outputs, reputational damage, regulatory scrutiny, and competition from other AI providers including hyperscalers and emerging companies.",
        "ticker": "MSFT",
        "category": "risk",
    },
    {
        "question": "What cybersecurity risks does Microsoft highlight?",
        "ground_truth": "Microsoft highlights risks from cyberattacks on its infrastructure, vulnerabilities in its software products, nation-state sponsored attacks, and the risk that its security implementations may be insufficient to protect customer data.",
        "ticker": "MSFT",
        "category": "risk",
    },

    # Cross-company comparison
    {
        "question": "Which company spent more on R&D in 2025, Apple or Microsoft?",
        "ground_truth": "Microsoft spent significantly more on R&D (~$33 billion) compared to Apple ($29,915 million) in fiscal year 2025.",
        "ticker": "BOTH",
        "category": "comparison",
    },
    {
        "question": "What do both Apple and Microsoft identify as a significant risk related to government regulation?",
        "ground_truth": "Both companies identify regulatory risks related to data privacy, antitrust scrutiny, and government policies that could restrict their business operations or impose additional compliance costs.",
        "ticker": "BOTH",
        "category": "comparison",
    },
]

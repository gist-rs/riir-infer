//! Emit a chat-shaped token file for the MTP-head acceptance re-read
//! (Bench 917 follow-up 2: the raw-text free-stream degeneracy is a
//! prompt-DISTRIBUTION artifact; the serving prompts are chat-shaped, and
//! the head's numbers there are the real scoreboard row).
//!
//! Writes raw u32-LE tokens (the `read_tokens` format the acceptance
//! harness consumes) for a multi-turn conversation transcript rendered in
//! the SERVING chat shape — `dist_stage::chat_turn`'s exact template:
//!
//! ```text
//! <|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n
//! ```
//!
//! alternating user/assistant turns; the file ends mid-assistant-turn so a
//! 768-token window reads like a live serving KV prefix. The transcript is
//! hand-written (support / classification / arithmetic / code / analysis
//! turns — the repo's serving prompt precedents), NOT model-generated: this
//! is a distribution re-read, the assistant replies only need to be
//! distributed like real assistant output, and the provenance must be
//! self-describing.
//!
//! ```text
//! cargo run --release -p riir-infer-dist --example mtp_chat_tokens -- \
//!   --model ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!   --out ../riir-train/data/bonsai_mtp/tokens.chat.tokens
//! ```

use std::path::PathBuf;

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::tokenizer::BpeTokenizer;

/// (user, assistant) turn pairs — serving-shaped, deliberately varied in
/// length and register so no single turn dominates a 768-token window.
const TURNS: &[(&str, &str)] = &[
    (
        "The customer was charged twice and wants a refund. What should I tell them?",
        "Acknowledge the duplicate charge, confirm the refund has been initiated, and give the 3-5 business day window. Offer the transaction ID lookup if they want it faster.",
    ),
    (
        "How do I reset my password?",
        "Go to Settings > Security > Reset password. You will receive an email with a link that is valid for 30 minutes.",
    ),
    (
        "Where is my order?",
        "Your order shipped yesterday and is in transit. The tracking number is in your shipping confirmation email; current estimate is delivery in 2 days.",
    ),
    ("What is 17 * 24? Give just the number.", "408"),
    ("What is 123 + 456? Give just the number.", "579"),
    (
        "A train travels 60 km per hour for 2.5 hours. How many km does it travel?",
        "150 km",
    ),
    (
        "Classify this ticket: 'API returns 500 intermittently under load but works locally every time.'",
        "Bug - intermittent server-side failure under concurrency. Likely a race or resource limit; needs load-env reproduction.",
    ),
    (
        "Write a Rust function that reverses a Vec<u32> in place.",
        "pub fn reverse(v: &mut Vec<u32>) {\n    let n = v.len();\n    for i in 0..n / 2 {\n        v.swap(i, n - 1 - i);\n    }\n}",
    ),
    (
        "What does this SQL do? SELECT user_id, COUNT(*) FROM orders GROUP BY user_id HAVING COUNT(*) > 5;",
        "It lists every user who has placed more than five orders, with their order counts. The GROUP BY collapses rows per user, and HAVING filters the aggregated groups.",
    ),
    (
        "Summarize: 'The committee reviewed three proposals and deferred a decision until the quarterly meeting, citing budget uncertainty and pending legal review of the largest bid.'",
        "The committee postponed all three proposals to the quarterly meeting, waiting on budget clarity and a legal review of the largest bid.",
    ),
    (
        "Is 'output gives the wrong number sometimes but only on weekends' enough to reproduce a bug?",
        "No. It names a symptom and a correlation but nothing reproducible: no input, environment, or version. Ask for timestamps, exact inputs, and logs around the failures.",
    ),
    ("Convert 45 minutes to seconds. Just the number.", "2700"),
    ("What is 15% of 80?", "12"),
    (
        "Explain idempotency in one sentence for an API doc.",
        "An idempotent endpoint produces the same result no matter how many times it is called with the same request, so clients can safely retry.",
    ),
    (
        "Which of these is a primary color: purple, green, red?",
        "Red",
    ),
    (
        "The deploy failed with 'image pull backoff'. What are the two most likely causes?",
        "A wrong or missing image tag (the registry has no such image), or a registry authentication problem (no valid credentials for a private image).",
    ),
    ("What year did the Berlin Wall fall?", "1989"),
    (
        "Name the four layers of the TCP/IP model in order.",
        "Application, transport, internet, and link.",
    ),
    (
        "Draft a one-line status update: migration is 80% done, on schedule.",
        "Migration is at 80% completion and tracking on schedule.",
    ),
    (
        "If a plan costs $12 per user per month and we have 40 users, what is the monthly cost?",
        "$480 per month",
    ),
    (
        "Does this log line indicate a client or server problem: 'connection reset by peer after 30s'? Explain briefly.",
        "Likely client-side or network: the peer closed the connection. A 30-second window before the reset suggests an idle timeout on an intermediary, not a crash.",
    ),
    ("What is the opposite of 'increase'?", "decrease"),
    (
        "Sort these by memory safety guarantees, strongest first: C, Rust, Python.",
        "Rust (compile-time ownership), Python (memory-safe runtime, no manual freeing), C (manual management, no runtime or compile-time safety net).",
    ),
    (
        "A cron job ran twice at 02:00. Give the first thing to check.",
        "Whether the scheduler's overlap lock (flock or equivalent) was in place — double fires are usually a missed lock or an over-long run, not duplicate crontab entries.",
    ),
    (
        "Rewrite to be polite: 'You broke the build again.'",
        "Heads up - the build is failing again; could you take a look when you have a moment?",
    ),
    ("What is 2^10?", "1024"),
    (
        "In one sentence, why does a rate limiter need a shared store across replicas?",
        "Without shared state each replica counts its own budget, so the effective limit multiplies by the replica count.",
    ),
    (
        "Ticket says 'app is slow'. Give two clarifying questions.",
        "Slow at what - startup, every request, or a specific operation? And slow since when - after which deploy or change?",
    ),
    (
        "What HTTP status should a successful POST that created a resource return?",
        "201 Created, with a Location header pointing at the new resource.",
    ),
    ("Complete the sequence: 2, 4, 8, 16, ...", "32"),
    (
        "Explain the difference between a process and a thread to a PM.",
        "A process is a program with its own private workspace; threads are workers inside that workspace sharing its memory. Threads are cheaper to create but can trip over each other; processes are isolated but heavier.",
    ),
    (
        "What does the acronym WAL stand for in databases, and what is it for?",
        "Write-ahead log: a durable append-only record of changes written before the data pages, so the database can recover committed work after a crash.",
    ),
    (
        "Estimate: how many requests per second can a single modest web server handle for 1ms of work per request? Ballpark.",
        "Order of thousands: at 1ms of pure work, the ceiling is roughly 1000 sequential requests per second per core, more with concurrency on I/O-bound work.",
    ),
    (
        "Which is cheaper for 10,000 lookups per second on a fixed key set: a HashMap rebuild per minute or a perfect-hash table compiled in? One sentence why.",
        "The compiled perfect-hash table: it removes the rebuild cost and the per-lookup probe variance, trading compile-time key knowledge for a single predictable compare.",
    ),
    (
        "The user's session expired mid-form. What is the least annoying recovery?",
        "Persist the draft locally, re-authenticate silently if possible, and restore the form with a 'session refreshed' note instead of discarding their input.",
    ),
    ("What is the capital of Australia?", "Canberra"),
    (
        "Give the git command to undo the last commit but keep the changes staged.",
        "git reset --soft HEAD~1",
    ),
    (
        "A customer asks if we are SOC 2 compliant. What may you say?",
        "Confirm the current audit status exactly as the compliance page states it, offer the report under NDA through sales, and do not improvise scope details.",
    ),
];

/// Serve-shaped traffic repeats INTENTS with different instances. Tile the
/// base turn set `reps` times; `rep` indexes a small parametric variation on
/// the numeric turns (the same question with different values, the way real
/// sessions arrive) so no two tiles are byte-identical. Substitution is by
/// turn index — never substring replacement over free text.
fn tile_turns(reps: usize) -> Vec<(String, String)> {
    // Indices within TURNS of the numeric turns + their value/answer fns.
    const MUL: usize = 3; // "17 * 24"
    const ADD: usize = 4; // "123 + 456"
    const TRAIN: usize = 5; // 60 km/h for 2.5 h
    const SECS: usize = 11; // 45 minutes
    const PCT: usize = 12; // 15% of 80
    const COST: usize = 19; // $12 * 40 users
    const POW: usize = 25; // 2^10
    let mut out = Vec::with_capacity(TURNS.len() * reps);
    for rep in 0..reps {
        for (i, (u, m)) in TURNS.iter().enumerate() {
            let (user, answer) = match i {
                MUL => {
                    let (a, b) = (17 + rep as i64, 24 + rep as i64 * 2);
                    (
                        format!("What is {a} * {b}? Give just the number."),
                        (a * b).to_string(),
                    )
                }
                ADD => {
                    let (a, b) = (123 + rep as i64 * 10, 456 + rep as i64 * 100);
                    (
                        format!("What is {a} + {b}? Give just the number."),
                        (a + b).to_string(),
                    )
                }
                TRAIN => {
                    let v = 60 + rep as i64 * 5;
                    (
                        format!(
                            "A train travels {v} km per hour for 2.5 hours. How many km does it travel?"
                        ),
                        format!("{} km", v * 2 + v / 2),
                    )
                }
                SECS => {
                    let m = 45 + rep as i64 * 6;
                    (
                        format!("Convert {m} minutes to seconds. Just the number."),
                        (m * 60).to_string(),
                    )
                }
                PCT => {
                    let n = 80 + rep as i64 * 20;
                    (format!("What is 15% of {n}?"), (n * 15 / 100).to_string())
                }
                COST => {
                    let n = 40 + rep as i64 * 5;
                    (
                        format!(
                            "If a plan costs $12 per user per month and we have {n} users, what is the monthly cost?"
                        ),
                        format!("${}", 12 * n),
                    )
                }
                POW => {
                    let e = 10 + rep as i64;
                    (format!("What is 2^{e}?"), format!("{}", 1i64 << e))
                }
                _ => ((*u).to_string(), (*m).to_string()),
            };
            out.push((user, answer));
        }
    }
    out
}

fn chat_turn(user: &str, assistant: &str) -> String {
    // dist_stage::chat_turn's exact serving template, extended with the
    // assistant reply so the transcript alternates like a live conversation.
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>\n")
}

fn main() {
    let mut model = PathBuf::from("../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf");
    let mut out = PathBuf::from("../riir-train/data/bonsai_mtp/tokens.chat.tokens");
    let mut reps: usize = 1;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = PathBuf::from(args.next().expect("--model value")),
            "--out" => out = PathBuf::from(args.next().expect("--out value")),
            "--reps" => reps = args.next().expect("--reps value").parse().expect("usize"),
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }
    let t0 = std::time::Instant::now();
    let gguf = GgufFile::open(&model).unwrap_or_else(|e| {
        eprintln!("open {}: {e}", model.display());
        std::process::exit(1);
    });
    let tok = BpeTokenizer::from_gguf(&gguf).unwrap_or_else(|e| {
        eprintln!("tokenizer: {e}");
        std::process::exit(1);
    });
    drop(gguf);

    let mut all: Vec<u32> = Vec::with_capacity(64 * 1024);
    // BOS once at the head of the transcript.
    all.push(tok.bos_id() as u32);
    let turns = tile_turns(reps);
    for (u, a) in &turns {
        let text = chat_turn(u, a);
        let ids = tok.encode(&text);
        all.extend(ids.into_iter().map(|t| t as u32));
    }
    std::fs::write(
        &out,
        all.iter()
            .flat_map(|t| t.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .unwrap_or_else(|e| {
        eprintln!("write {}: {e}", out.display());
        std::process::exit(1);
    });
    println!(
        "# wrote {} tokens ({} turns, {reps} reps) to {} ({:.1}s)",
        all.len(),
        turns.len(),
        out.display(),
        t0.elapsed().as_secs_f32()
    );
}

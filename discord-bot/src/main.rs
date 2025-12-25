use std::env;

use serenity::async_trait;
use serenity::model::channel::Message;
use serenity::model::gateway::Ready;
use serenity::prelude::*;

struct Handler;

#[async_trait]
impl EventHandler for Handler {
    // Set a handler for the 'message' event. This is called whenever a new message is recieved.
    //
    // Event handlers are dispatched through a threadpool,
    // so multiple events can be dispatched simultaneously.
    async fn message(&self, context: Context, message: Message) {
        if message.content == "!ping" {
            // Sending a message can fail, log to stdout with a description.
            if let Err(why) = message.channel_id.say(&context.http, "Fizzbot").await {
                println!("Error sending message: {why:?}");
            }
        }
    }

    // Set a handler to be called on the 'ready' event. This is called when a shard is booted, and
    // a READY paylod is sent by Discord. This payload contains data like the current user's guild
    // Ids, current user data, private channels, and more.
    //
    // In this case, just print what the current user's username is.
    async fn ready(&self, _: Context, ready: Ready) {
        println!("{} is connected!", ready.user.name);
    }
}

#[tokio::main]
async fn main() {
    // Configure with bot token environment variable, if no token set load from .env
    dotenvy::dotenv().ok();
    let token = env::var("DISCORD_TOKEN").expect("Expected a token in the environment.");
    // Set gateway intents, which decides what the bot will be notified about
    let intents = GatewayIntents::GUILD_MESSAGES | GatewayIntents::MESSAGE_CONTENT;

    // Create a new instance of the Client, logging in as a bot.
    // This will automatically prepend your bot token with "Bot ",
    // which is a requirement by Discord.
    let mut client = Client::builder(&token, intents)
        .event_handler(Handler)
        .await
        .expect("Error creating client");

    // Finally, start a single shard, and start listening to events.
    //
    // Shards will automatically attempt to reconnect, and will perform exponential backoff until
    // it reconnects.
    if let Err(why) = client.start().await {
        println!("Client error: {why:?}");
    }
}

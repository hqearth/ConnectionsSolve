import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import tensorflow as tf
tf.get_logger().setLevel('FATAL')

import numpy as np
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, BatchNormalization
from collections import deque
import matplotlib.pyplot as plt
from scipy import signal

#one hot encode states
def one_hot(ind, size):
    vector = np.zeros(size)
    vector[ind] = 1
    return vector

class ConnectionsGame:
    def __init__(self, words, groups):
        self.start_words = words
        self.words = list(words)
        self.word_ind = {word: i for i, word in enumerate(words)}
        self.assign_group = {word: group for word, group in zip(words, groups)}
        self.state_size = len(words)
        self.action_size = len(words) + 1
        self.state = []
        self.guess_ct = 0
        self.bad_guess_ct = 0
        self.num_guesses = 5
        self.prev_guesses = set()

    def reset(self):
        self.words = list(self.start_words)
        self.state = []
        self.guess_ct = 0
        self.bad_guess_ct = 0
        self.prev_guesses = set()
        return self.update()

    def update(self):
        indices = [self.word_ind[word] for word in self.state if word in self.words]
        return one_hot(indices, self.state_size)

    def step(self, action):
        done = False
        reward = 0

        if action == self.action_size - 1:  #guess
            state_tup = tuple(sorted(self.state))
            #penalty if repeat guess
            if state_tup in self.prev_guesses:
                repeat_penalty = -2.5
                reward = repeat_penalty
            else:
                self.prev_guesses.add(state_tup)
                repeat_penalty = 0

            #reward if complete group
            if len(self.state) == 4 and len(set(self.state)) == 4:
                self.guess_ct += 1
                group_cts = {}

                for word in self.state:
                    group = self.assign_group[word]
                    group_cts[group] = group_cts.get(group, 0) + 1

                max_group = max(group_cts.values(), default=0)
                if max_group == 4:
                    reward = 5
                    self.words = [word for word in self.words if word not in self.state]  #remove correctly guessed words
                    print("****** HIT A CORRECT GROUP ******")
                    print(f"Words remaining after removal: {self.words}")
                
                #give less rewards for partially correct groups
                elif max_group == 3:
                    reward = 1.5 + repeat_penalty
                    self.bad_guess_ct += 1
                elif max_group == 2:
                    reward = -0.5 + repeat_penalty
                    self.bad_guess_ct += 1
                else:
                    reward = -1 + repeat_penalty
                    self.bad_guess_ct += 1

                print(f"Current state: {self.state}, Number of words: {len(self.words)}, R: {reward}, State size: {self.state_size}, Guesses: {self.bad_guess_ct}")
                done = self.bad_guess_ct >= self.num_guesses

                if len(self.words) <= 4:
                    done = True
                    print(f"Ending with {self.words} in pool with {self.bad_guess_ct} guesses used")
                    print("REACHED SOLUTION")

                self.state = []  # Reset current state
            else:
                reward = -2


            updated_state = self.update()
            return updated_state, reward, done, self.bad_guess_ct

        #adding words and not guessing
        else:
            if action < len(self.start_words) and self.start_words[action] in self.words and len(self.state) < 4:
                picked_word = self.start_words[action]
                #reward adding good words
                if picked_word not in self.state:
                    self.state.append(picked_word)
                    reward = 0.5 * sum(1 for other in self.state if self.assign_group[other] == self.assign_group[picked_word] and other != picked_word)
                else:
                    reward = -0.25
            else:
                reward = -0.1

        updated_state = self.update()
        return updated_state, reward, done, self.bad_guess_ct




def create_model(input_size, output_size):
    #testing model
    model = Sequential([
        # Dense(128, activation='elu', input_dim=input_size),
        # BatchNormalization(),
        # Dropout(0.1),
        # Dense(128, activation='relu'),
        # BatchNormalization(),
        # Dropout(0.15),
        # Dense(64, activation='relu'),
        # Dense(output_size, activation='linear')
        Dense(128, activation='elu', input_dim=input_size),
        Dropout(0.2),
        Dense(128, activation='relu'),
        #BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(output_size, activation='linear')
    ])
    return model

#double Q learning
def dqn_loss(model, target_model, states, actions, rewards, next_states, dones, gamma=0.99):
    total_loss = 0.0
    batch_size = states.shape[0]

    q_values = model(states, training=True)
    next_q_values_online = model(next_states, training=False)
    next_q_values_target = target_model(next_states, training=False)

    for i in range(batch_size):
        action = actions[i]
        reward = rewards[i]
        done = dones[i]
        check_s = next_states[i]

        q_pred = q_values[i, action]

        if done or np.mean(check_s) == 0:
            q_target = reward
        else:
            #pick best act
            best_action = np.argmax(next_q_values_online[i])
            #eval with target
            q_target = reward + gamma * next_q_values_target[i, best_action]

        squared_diff = (q_pred - q_target) ** 2
        total_loss += squared_diff

    average_loss = total_loss / batch_size
    return average_loss


def dqn_loss_sing(model, target_model, states, actions, rewards, next_states, dones, gamma=0.99):
    total_loss = 0.0
    batch_size = states.shape[0]

    q_values = model(states, training=True)
    next_q_values = target_model(next_states, training=False)

    for i in range(batch_size):
        action = actions[i]
        reward = rewards[i]
        done = dones[i]
        q_pred = q_values[i, action]

        if done:
            q_target = reward
        else:
            q_target = reward + gamma * np.max(next_q_values[i])

        squared_diff = (q_pred - q_target) ** 2
        total_loss += squared_diff

    average_loss = total_loss / batch_size
    return average_loss

def train_dqn(env, episodes, batch_size, model_dir='model_checkpoints'):
    model = create_model(env.state_size, env.action_size)
    target_model = create_model(env.state_size, env.action_size)
    target_model.set_weights(model.get_weights())
    buffer = deque(maxlen=100000)
    epsilon = 0.5
    track_r = []
    track_l = []
    losses = []
    total_steps = 0

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=0.6)
    #make dir to save models
    os.makedirs(model_dir, exist_ok=True)
    track_tries = []
    exploring = []
    exploiting = []
    avg_r = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        step_count = 0  # To monitor how many steps are taken in each episode
        total_reward = 0
        ep_loss = []
        expr_track = 0
        expt_track = 0

        while not done:
            #eps greedy
            if random.random() < epsilon or (len(env.words) <= 8 and np.mean(state) == 0):
                valid_actions = [i for i in range(len(env.words))] + [env.action_size - 1]
                action = random.choice(valid_actions)
                expr_track += 1
            else:
                action_probs = model.predict(np.array([state]), verbose=0).flatten()
                action_probs = [action_probs[i] if i < len(env.words) or i == env.action_size - 1 else 0 for i in range(env.action_size)]
                action = np.argmax(action_probs)
                expt_track += 1

            next_state, reward, done, tries = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += 0.99**(step_count)*reward
            step_count += 1
            total_steps += 1

            #train every 20 steps
            if len(buffer) >= batch_size and total_steps % 20 == 0:
                batch = random.sample(buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                next_states = np.array(next_states)
                dones = np.array(dones)

                #preventing error with tape.gradient
                if np.mean(states) == 0:
                    loss = dqn_loss(model, target_model, states, actions, rewards, next_states, dones)
                    grads = [tf.zeros_like(variable) for variable in model.trainable_variables]

                else:
                    with tf.GradientTape() as tape:
                        loss = dqn_loss(model, target_model, states, actions, rewards, next_states, dones)
                    #trying to prevent rare exploding computation time w/ None vals in grads
                    grads = tape.gradient(loss, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                    grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)

                if grads is not None:
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                ep_loss.append(loss)

            #save model when solves problem
            if done and len(env.words) <= 4:
                model_path = os.path.join(model_dir, f'model_ep{episode}_guesses{env.bad_guess_ct}.h5')
                print(f"Saving model to {model_path}")
                model.save(model_path)
                print(f"Saved model to {model_path}")
                print(f"Number of tries: {tries}")
                track_tries.append(tries)
                
            #target update
            if total_steps % 500 == 0:
                target_model.set_weights(model.get_weights())

        #decay epsilon
        epsilon = max(0.05, epsilon*0.998)
        
        #for plotting
        this_loss = np.mean(ep_loss)
        track_l.append(this_loss)
        losses.append(np.mean(track_l))
        track_r.append(total_reward)
        avg_r.append(np.mean(track_r))
        exploring.append(expr_track)
        exploiting.append(expt_track)

        print(f'Episode: {episode}, Epsilon: {epsilon}, Total Steps: {total_steps}, Total R: {total_reward}, Avg R: {np.mean(track_r)}, Avg Loss: {np.mean(ep_loss) if ep_loss else 0}, [expr, expt]: {expr_track}, {expt_track}')
        #saving plotting values in case of error
        if episode % 20 == 0:
            f = open('vals.txt', "w")
            f.write(f"{losses}, {avg_r}, {track_tries}, {exploring}, {exploiting}")

    return model, losses, avg_r, track_tries, exploring, exploiting


#starting words and groups
words = ['apple', 'banana', 'grape', 'pear', 'chair', 'table', 'desk', 'sofa', 'blue', 'yellow', 'green', 'red', 'denver', 'boulder', 'vail', 'greeley']
groups = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
env = ConnectionsGame(words, groups)
model, losses, rewards, tries, exploring, exploiting = train_dqn(env, 750, 32)
model.save('finalmodel.keras')

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.title('Training Losses')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(rewards, label='Rewards', color='r')
plt.title('Training Rewards')
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('rewards.png')

plt.figure()
plt.plot(tries)
plt.title('Number of Incorrect Guesses for Each Success')
plt.xlabel('Episode')
plt.ylabel('Number of Guesses')
plt.savefig('guesses.png')

plt.figure()
plt.plot(exploring, label = "Exploration")
plt.plot(exploiting, label = "Exploitation")
plt.title('Exploration vs. Exploitation over Episodes')
plt.xlabel('Episode')
plt.ylabel('Actions Taken')
plt.legend()
plt.savefig('explore.png')

window = np.ones(10)/10

smoothed_expr = np.convolve(exploring, window, mode = 'valid')
smoothed_expt = np.convolve(exploiting, window, mode = 'valid')
plt.figure()
plt.plot(smoothed_expr, label = "Exploration")
plt.plot(smoothed_expt, label = "Exploitation")
plt.title('Exploration vs. Exploitation over Episodes')
plt.xlabel('Episode')
plt.ylabel('Actions Taken')
plt.legend()
plt.savefig('smoothed.png')
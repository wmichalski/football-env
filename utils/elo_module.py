import math

class EloModule:
    def __init__(self):
        self.k = 32

    def get_win_probability(self, e1, e2):
        q_e2 = math.pow(10, e2/400)
        q_e1 = math.pow(10, e1/400)

        e2_win_prob = q_e2/(q_e2 + q_e1)
        e1_win_prob = q_e1/(q_e1 + q_e2)

        return e1_win_prob, e2_win_prob

    def update_scores(self, elo_p1, elo_p2, p1_score):
        # p1_score is percentage of games won by p1 - <0,1>

        e1_win_prob, e2_win_prob = self.get_win_probability(elo_p1, elo_p2)

        elo_p1 += self.k * (p1_score - e1_win_prob)
        elo_p2 += self.k * ((1-p1_score) - e2_win_prob)

        has_p1_won = True if (p1_score - e1_win_prob) > 0 else False

        return elo_p1, elo_p2, has_p1_won, abs(self.k * (p1_score - e1_win_prob))

    def process_match(self, player1, player2, p1_wins, draws, p2_wins):
        actual_p1_score = (p1_wins + 0.5 * draws) / (p1_wins + draws + p2_wins)

        return self.update_scores(player1.elo, player2.elo, actual_p1_score)

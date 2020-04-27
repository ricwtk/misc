let vm = new Vue({
  el: '#app',
  vuetify: new Vuetify(),
  data: {
    cave: {
      size: [4,4],
      start: [0,0],
      wumpus: {
        position: [3,3],
        dead: false
      },
      gold: [2,2],
      pits: [[1,2],[3,2]],
      archer: {
        position: [0,0],
        facing: 'e',
        arrow: true,
        dead: false
      },
      explored: [[0,0]]
    },
    game_over: false,
    points: 0,
    m_unit: 100,
    logs: [],
    show_answer: false,
    dead_wumpus_indication: {
      show: false,
      position: [0,0]
    },
    info_interface: false
  },
  mounted: function () {
    document.onkeyup = (e) => {
      if (e.which == 37) {
        this.turn_left();
      } else if (e.which == 39) {
        this.turn_right();
      } else if (e.which == 38) {
        this.move_forward();
      } else if (e.which == 32) {
        this.shoot_arrow();
      }
    };
    this.regenerate_cave();
  },
  watch: {
    game_over: function (val) {
      if (val) {
        this.$refs.winWord.beginElement();
        this.$refs.loseWord.beginElement();
      }
    }
  },
  computed: {
    cave_view_box: function () {
      return `0 0 ${(this.cave.size[0]+2)*this.m_unit} ${(this.cave.size[1]+2)*this.m_unit}`;
    },
  },
  methods: {
    get_arr_str: function (arr) { return arr.map(v => JSON.stringify(v)); },
    get_y_top: function (r) { return ( this.cave.size[1] - r ) * this.m_unit; },
    get_y_bot: function (r) { return ( this.cave.size[1] - r ) * this.m_unit + (this.m_unit - 1); },
    get_y_center: function (r) { return ( this.cave.size[1] - r ) * this.m_unit + .5 * this.m_unit; },
    get_x_left: function (c) { return ( c + 1 ) * this.m_unit; },
    get_x_right: function (c) { return ( c + 1 ) * this.m_unit + (this.m_unit - 1); },
    get_x_center: function (c) { return ( c + 1 ) * this.m_unit + .5 * this.m_unit; },
    is_a_pit: function (c,r) { return this.get_arr_str(this.cave.pits).includes(JSON.stringify([c,r])); },
    is_wumpus: function (c,r) { return JSON.stringify(this.cave.wumpus.position) == JSON.stringify([c,r]); },
    is_gold: function (c,r) { return JSON.stringify(this.cave.gold) == JSON.stringify([c,r]); },
    is_explored: function (c,r) { return this.get_arr_str(this.cave.explored).includes(JSON.stringify([c,r])); },
    move_forward: function () {
      if (!this.game_over) {
        let idx_to_change = 'we'.includes(this.cave.archer.facing) ? 0 : 1;
        let increment = 'ws'.includes(this.cave.archer.facing) ? -1 : 1;
        let new_value = this.cave.archer.position[idx_to_change] + increment;
        if (new_value < this.cave.size[idx_to_change] && new_value > -1) {
          this.cave.archer.position.splice(idx_to_change, 1, new_value);
          this.points += -1;
        }
        if (!this.is_explored(this.cave.archer.position[0], this.cave.archer.position[1])) {
          this.cave.explored.push(JSON.parse(JSON.stringify(this.cave.archer.position)));
        }
        this.logs.push({
          type: "notification",
          text: `the archer moves to ${this.cave.archer.position}`
        });
        // death check (pit, wumpus)
        if (this.is_a_pit(this.cave.archer.position[0],this.cave.archer.position[1]) || (!this.cave.wumpus.dead && this.is_wumpus(this.cave.archer.position[0],this.cave.archer.position[1])) ) {
          this.cave.archer.dead = true;
          this.cave.archer.facing = 'e';
          this.points += -1000;
          this.game_over = true;
          this.logs.push({
            type: "notification",
            text: this.is_a_pit(this.cave.archer.position[0],this.cave.archer.position[1]) ? "Oops! it's a pit, the archer dies" : "Oops! the archer is eaten by wumpus"
          });
        }
        if (this.is_gold(this.cave.archer.position[0],this.cave.archer.position[1])) {
          this.points += 1000;
          this.game_over = true;
          this.logs.push({
            type: "notification",
            text: "YAY! the archer got the gold!"
          });
        }
      }
    },
    turn_left: function () {
      if (!this.game_over) {
        let seq = 'nwse';
        let idx = seq.indexOf(this.cave.archer.facing);
        this.cave.archer.facing = seq[(idx + 1)%seq.length];
        this.points += -1;
        this.logs.push({
          type: "notification",
          text: "the archer turns left"
        });
      }
    },
    turn_right: function () {
      if (!this.game_over) {
        let seq = 'nesw';
        let idx = seq.indexOf(this.cave.archer.facing);
        this.cave.archer.facing = seq[(idx + 1)%seq.length];
        this.points += -1;
        this.logs.push({
          type: "notification",
          text: "the archer turns right"
        });
      }
    },
    shoot_arrow: function () {
      if (this.cave.archer.arrow && !this.game_over) {
        let idx_to_check = 'we'.includes(this.cave.archer.facing) ? 1 : 0;
        this.logs.push({
          type: "notification",
          text: "arrow is shot"
        });
        if (this.cave.wumpus.position[idx_to_check] == this.cave.archer.position[idx_to_check]) {
          if ('ne'.includes(this.cave.archer.facing)) {
            this.cave.wumpus.dead = this.cave.wumpus.position[1-idx_to_check] > this.cave.archer.position[1-idx_to_check];
          } else {
            this.cave.wumpus.dead = this.cave.wumpus.position[1-idx_to_check] < this.cave.archer.position[1-idx_to_check];
          }
        }
        if (this.cave.wumpus.dead) { 
          this.dead_wumpus_indication.show = true;
          this.dead_wumpus_indication.position[idx_to_check] = this.cave.wumpus.position[idx_to_check];
          this.dead_wumpus_indication.position[1-idx_to_check] = this.cave.size[1-idx_to_check];
          this.logs.push({
            type: "notification",
            text: "YAY! wumpus is dead"
          });
        } else {
          this.logs.push({
            type: "notification",
            text: "Oops, wumpus is still alive"
          });
        }
        this.points += -10;
        this.cave.archer.arrow = false;
      }
    },
    restart_cave: function () {
      this.cave.archer.position = JSON.parse(JSON.stringify(this.cave.start));
      this.cave.archer.facing = 'e';
      this.cave.archer.arrow = true;
      this.cave.archer.dead = false;
      this.cave.wumpus.dead = false;
      this.cave.explored = [[0,0]];
      this.points = 0;
      this.show_answer = false;
      this.game_over = false;
      this.dead_wumpus_indication.show = false;
      this.dead_wumpus_indication.position = [0,0];
      this.logs.push({
        type: "notification",
        text: "cave status is reset"
      });
    },
    regenerate_cave: function () {
      let options = [0,1,2,3].flatMap(r => [0,1,2,3].map(c => [c,r]));
      options = options.filter( opt => !this.get_arr_str([[0,0],[0,1],[1,0]]).includes(JSON.stringify(opt)) );
      this.cave.wumpus.position = JSON.parse(JSON.stringify(options[Math.floor(Math.random() * options.length)]));
      options = options.filter( opt => !this.get_arr_str([this.cave.wumpus.position]).includes(JSON.stringify(opt)) );
      console.log(this.cave.wumpus.position);
      this.cave.gold = JSON.parse(JSON.stringify(options[Math.floor(Math.random() * options.length)]));
      options = options.filter( opt => !this.get_arr_str([this.cave.gold]).includes(JSON.stringify(opt)) );
      this.cave.pits = [];
      options.forEach( opt => {
        if (Math.random() <= 0.2) {
          this.cave.pits.push(JSON.parse(JSON.stringify(opt)));
        }
      } );
      this.logs.push({
        type: "notification",
        text: "cave is regenerated"
      });
      this.restart_cave();
    },
    select_cell: function (c,r) {
      console.log(c,r);
    }
  }
});
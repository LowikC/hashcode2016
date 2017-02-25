import numpy as np
from collections import defaultdict
from PriorityDict import PriorityDict


class EndPoint:
    def __init__(self, datacenter_latency, cache_ids=None, cache_latencies=None):
        self.datacenter_latency = datacenter_latency
        self.cache_ids = cache_ids if cache_ids else []
        self.cache_latencies = cache_latencies if cache_latencies else []

    @property
    def size(self):
        return len(self.cache_ids)

    def add(self, cid, clat):
        # Don't consider caches that have a latency higher than the one of the datacenter
        # there are useless.
        if clat < self.datacenter_latency:
            self.cache_ids.append(cid)
            self.cache_latencies.append(clat)


class CacheSolver:
    def __init__(self, filename, weight=1.0):
        with open(filename, "r") as in_file:
            self.weight = weight
            self.video_sizes = None
            self.n_cids = 0
            self.cache_size = 0
            self.endpoints = []
            self.requests = None
            self.total_n_requests = 0
            self.read_input(in_file)

            self.requests_with_latency = None
            self.eids_per_cid = None
            self.latency_eid_cid = None
            self.total_time_gain = 0

            self.remaining_capacity = None
            self.current_solution = None
            self.vid_cid_gains = None

    def read_input(self, input_file):
        read_line = lambda _: [int(x) for x in input_file.readline().strip().split()]

        n_vids, n_eids, n_reqs, self.n_cids, self.cache_size = read_line(0)

        self.video_sizes = read_line(0)
        assert(n_vids == len(self.video_sizes))

        for _ in range(n_eids):
            lat, nchaches = read_line(0)
            curr = EndPoint(lat)
            for _ in range(nchaches):
                curr.add(*read_line(0))
            self.endpoints.append(curr)

        # Read requests and aggregate by (vid, eid)
        self.requests = defaultdict(int)
        self.total_n_requests = 0
        for _ in range(n_reqs):
            vid, eid, n = read_line(0)
            self.requests[(vid, eid)] += n
            self.total_n_requests += n

    def initialize(self):
        # Keep track of the remaining capacity in each cache.
        self.remaining_capacity = [self.cache_size for _ in range(self.n_cids)]

        # Keep track of the best latency for each request (indexed by (vid, eid))
        self.requests_with_latency = dict()
        for (vid, eid), n in self.requests.items():
            self.requests_with_latency[(vid, eid)] = (n, self.endpoints[eid].datacenter_latency)

        # set of endpoints connected to each cache.
        self.eids_per_cid = defaultdict(set)
        for eid, e in enumerate(self.endpoints):
            for cid in e.cache_ids:
                self.eids_per_cid[cid].add(eid)

        # Latency for every (endpoint, cache)
        self.latency_eid_cid = dict()
        for eid, endpoint in enumerate(self.endpoints):
            for cid, latency in zip(endpoint.cache_ids, endpoint.cache_latencies):
                self.latency_eid_cid[eid, cid] = latency

        # Keep track of the gain in score if we add (vid, cid) to the current solution
        self.vid_cid_gains = PriorityDict()
        for vid in range(len(self.video_sizes)):
            for cid in range(self.n_cids):
                if self.enough_capacity(vid, cid):
                    gain = self.compute_gain(vid, cid)
                    if gain > 0:
                        self.vid_cid_gains[(vid, cid)] = -gain

        print("There are {} (vid, cid)".format(len(self.vid_cid_gains)))
        # The solution is a dict cache_id -> [set of video_id in this cache]
        self.current_solution = defaultdict(set)

    def enough_capacity(self, vid, cid):
        return self.remaining_capacity[cid] - self.video_sizes[vid] >= 0

    def run(self):
        self.initialize()

        while 1:
            best = self.get_max_valid_gain()
            if best is not None:
                self.update(best)
            else:
                # Can't find a valid pair, we're done
                break

        return self.current_solution

    def compute_gain(self, vid, cid):
        """
        Compute the gain in score if we had (vid, cid) to the current solution.

        :return: The gain in time.
        """
        connected_endpoints = self.eids_per_cid[cid]
        total_time_gain = 0
        for eid in connected_endpoints:
            if (vid, eid) in self.requests_with_latency:
                n, current_latency = self.requests_with_latency[(vid, eid)]
                current_time = n * current_latency
                time_with_cid = self.latency_eid_cid[(eid, cid)] * n
                time_gain = max(0, current_time - time_with_cid)
                total_time_gain += time_gain
        return total_time_gain/pow(self.video_sizes[vid], self.weight)

    def get_max_valid_gain(self):
        while self.vid_cid_gains:
            vid, cid = self.vid_cid_gains.pop_smallest()
            if self.enough_capacity(vid, cid):
                return vid, cid

        return None

    def update_capacity(self, vid, cid):
        self.remaining_capacity[cid] -= self.video_sizes[vid]
        assert(self.remaining_capacity[cid] >= 0)

    def update_request_latency(self, vid, cid):
        """
        Update the best latency of all requests, knowing we add (vid, cid) to the solution.

        We just need to update the requests (rvid, reid) meeting the 2 conditions:
         - rvid = vid
         - cid is connected to reid
        """
        impacted_eids = self.eids_per_cid[cid]
        for eid in impacted_eids:
            if (vid, eid) in self.requests_with_latency:
                n, current_latency = self.requests_with_latency[(vid, eid)]
                new_latency = min(current_latency, self.latency_eid_cid[(eid, cid)])
                self.requests_with_latency[(vid, eid)] = (n, new_latency)

    def update_gains(self, vid, cid):
        """
        Update the gain for all tuple (cvid, ccid) knowing we added (vid, cid) to the solution.
        We don't need to update all the gains, only those meeting:
        - cvid == vid (other videos are not impacted)
        - ccid is a cache of an endpoint connected to cid.
        """
        impacted_caches = set()
        impacted_eids = self.eids_per_cid[cid]
        for eid in impacted_eids:
            for cid in self.endpoints[eid].cache_ids:
                impacted_caches.add(cid)

        for c in impacted_caches:
            if (vid, c) in self.vid_cid_gains:
                self.vid_cid_gains[(vid, c)] = -self.compute_gain(vid, c)

    def update(self, best):
        vid, cid = best
        # add it to the current solution
        self.current_solution[cid].add(vid)

        self.update_capacity(vid, cid)
        self.update_request_latency(vid, cid)
        self.update_gains(vid, cid)

    def is_valid_solution(self, solution):
        # First, check capacity of caches
        for cid, vids in solution.items():
            size = np.sum([self.video_sizes[v_id] for v_id in vids])
            if size > self.cache_size:
                print("Cache {}, capacity exceeded {}/{}".format(cid, size, cache_size))
                return False
            # check unicity of videos in cache
            if len(vids) != len(set(vids)):
                print("Cache {}, presence of duplicates".format(cid))
                return False
        return True

    def get_score(self, solution):
        if not self.is_valid_solution(solution):
            return -1

        total_n_reqs = 0
        time_saved = 0

        for (vid, eid), n_reqs in self.requests.items():
            total_n_reqs += n_reqs
            # Check latency gain
            time_dataserver = self.endpoints[eid].datacenter_latency * n_reqs
            min_latency = self.endpoints[eid].datacenter_latency
            for cid in self.endpoints[eid].cache_ids:
                if vid in solution[cid]:
                    new_latency = self.latency_eid_cid[(eid, cid)]
                    min_latency = min(min_latency, new_latency)

            time_real = min_latency * n_reqs
            gain = time_dataserver - time_real
            time_saved += gain
        return int(time_saved * 1000 / total_n_reqs)

    @staticmethod
    def write(output_file, vids_per_cid):
        output_file.write("{n}\n".format(n=len(vids_per_cid)))
        for cid, videos in vids_per_cid.items():
            videos_str = " ".join([str(vid) for vid in videos])
            output_file.write("{cid} {videos}\n".format(cid=cid, videos=videos_str))

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        raise ValueError('Usage: Solver.py <input> [<weight>]')

    weight = 1.0 if len(sys.argv) < 3 else float(sys.argv[2])

    solver = CacheSolver(sys.argv[1], weight)

    solution = solver.run()
    score = solver.get_score(solution)
    if score < 0:
        raise Exception("Invalid solution")
    else:
        print("Score: {}".format(score))

    with open(sys.argv[1][:-3] + '.out', 'w') as out_file:
        solver.write(out_file, solution)


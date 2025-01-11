#ifndef TRANSPOSE_CORE_H
#define TRANSPOSE_CORE_H
#include "Mesh.h"

struct Photon3D
{
    Point pos{0, 0, 0};
    Vec3f dir{0, 0, 1};
    Scalar weight = 1.0f;
    Scalar max_z  = 0;
    Scalar Ps     = 0;
    int type      = 0;
    bool alive    = true;
    Index curPyramid;
    Index nextPyramid;
    Face nextFace;
};
enum class CollectType
{
    EMIT       = 0,
    COLLECT    = 1,
    OUTOFRANGE = 2,
    IGNORE     = -1
};
typedef struct resultType
{
    CollectType type = CollectType::IGNORE;
    Index pyramidIndex;
    Point pos;
    Vec3f dir;
    Scalar weight;
} resultType;
// 由于虚函数表的问题，在cuda上运行的不能直接使用虚函数
class DefaultCollectStrategy
{
   public:
    static constexpr int EMIT = 0;
    Kokkos::UnorderedMap<Index, CollectType, ExecSpace>& collect_map;
    DefaultCollectStrategy(Kokkos::UnorderedMap<Index, CollectType, ExecSpace>& collect_map) : collect_map(collect_map)
    {
    }
    KOKKOS_FUNCTION
    CollectType GetCollectType(Index pyIndex) const
    {
        Kokkos::printf("%s\n", __LINE__);
        return collect_map.value_at(pyIndex);
    }
};
class transpose_core
{
    // 单个光子的传输过程，包括发射、传输、吸收、散射、收集
    // 必须只含有POD数据和引用类型（不能有View类型）

   public:
    const TetMesh& m_mesh;
    Photon3D m_photon;
    resultType result;
    const RandPoolType& rand_pool;
    const DefaultCollectStrategy& m_collectStrategy;
    KOKKOS_INLINE_FUNCTION
    transpose_core(const TetMesh& mesh, const DefaultCollectStrategy& collectStrategy, const RandPoolType& rand_pool)
        : m_mesh(mesh), m_collectStrategy(collectStrategy), m_photon(), rand_pool(rand_pool)
    {
    }
    KOKKOS_INLINE_FUNCTION
    void run(bool log = false)
    {
        set_log(log);
        Emit();
        int i = MAX_ITER;
        CheckInit();
        while (m_photon.alive && i--)
        {
            Move();
            Roulette();
        }

        result.type = CollectType::IGNORE;
    }
    bool m_log = false;
    KOKKOS_INLINE_FUNCTION
    void set_log(bool log) { m_log = log; }
    template <typename... Args>
    KOKKOS_FORCEINLINE_FUNCTION void Printf(const char* format, Args... args)
    {
        if (!m_log) return;
        Kokkos::printf(format, args...);
    }
    template <typename... Args>
    KOKKOS_FORCEINLINE_FUNCTION void Printf_error(const char* format, Args... args)
    {
        Kokkos::printf("Error: ");
        Kokkos::printf(format, args...);
    }
    typedef struct Function_log_guard
    {
        transpose_core& m_core;
        const char* m_func_name;
        const char* m_file_name;
        int m_line_num;
        KOKKOS_INLINE_FUNCTION
        void print_photon()
        {
            m_core.Printf("m_photon.curPyramid: %d\n", m_core.m_photon.curPyramid);
            m_core.Printf("m_photon.nextPyramid: %d\n", m_core.m_photon.nextPyramid);
            m_core.Printf("m_mesh.pyramids.extent(0): %d\n", m_core.m_mesh.pyramids.extent(0));
            m_core.Printf("m_photon.alive: %d\n", m_core.m_photon.alive);
            m_core.Printf("m_photon.weight: %f\n", m_core.m_photon.weight);
            m_core.Printf("m_photon.max_z: %f\n", m_core.m_photon.max_z);
            m_core.Printf("m_photon.Ps: %f\n", m_core.m_photon.Ps);
            m_core.Printf("m_photon.type: %d\n", m_core.m_photon.type);
            m_core.Printf("m_photon.pos: %f %f %f\n", m_core.m_photon.pos.x, m_core.m_photon.pos.y,
                          m_core.m_photon.pos.z);
            m_core.Printf("m_photon.dir: %f %f %f\n", m_core.m_photon.dir.x, m_core.m_photon.dir.y,
                          m_core.m_photon.dir.z);
        }
        KOKKOS_FUNCTION
        Function_log_guard(transpose_core& core, const char* func_name, const char* file_name, int line_num)
            : m_core(core), m_func_name(func_name), m_file_name(file_name), m_line_num(line_num)
        {
            m_core.Printf("---------------------------START FUNCTION <%s>-------------------------------\n", func_name);
            m_core.Printf("> file: %s line: %d\n", file_name, line_num);
            print_photon();
            m_core.Printf("---------------------------START FUNCTION <%s>-------------------------------\n", func_name);
        }
        KOKKOS_FUNCTION
        ~Function_log_guard()
        {
            m_core.Printf("---------------------------END FUNCTION <%s>---------------------------------\n", m_func_name);
            print_photon();
            m_core.Printf("---------------------------END FUNCTION <%s>---------------------------------\n", m_func_name);
        }
    } Function_log_guard;
#define FUNCTION_LOG_GUARD Function_log_guard guard(*this, (const char*)__FUNCTION__, (const char*)__FILE__, __LINE__);
    KOKKOS_INLINE_FUNCTION
    void CheckInit()
    {
        FUNCTION_LOG_GUARD;
        Printf("m_photon.curPyramid: %d\n", m_photon.curPyramid);
        Printf("m_mesh.pyramids.extent(0): %d\n", m_mesh.pyramids.extent(0));
        KOKKOS_ASSERT(m_photon.curPyramid > 0);
        KOKKOS_ASSERT(m_photon.curPyramid < m_mesh.pyramids.extent(0));
        KOKKOS_ASSERT(m_photon.dir.norm() == 1);
    }
    KOKKOS_INLINE_FUNCTION
    Scalar GetRandom(Scalar lower = 0, Scalar upper = 1) { return rand_pool.get_state().drand(); }
    KOKKOS_INLINE_FUNCTION
    int FindCurPyramid()
    {
        for (int i = 0; i < m_mesh.pyramids.extent(0); i++)
        {
            if (m_mesh.pyramids[i].InPyramid(m_photon.pos))
            {
                return i;
            }
        }
        return -1;
    }
    KOKKOS_INLINE_FUNCTION
    bool Emit()
    {
        FUNCTION_LOG_GUARD;

        // emit a photon
        if (m_photon.curPyramid == -1)
        {
            Printf_error(
                "未指定初始位置所在pyramid, emit需要计算光子初始位置所在pyramid, 此操作会耗费大量时间, "
                "请尽量使用预设curPyramid\n");
            m_photon.curPyramid = FindCurPyramid();
        }
        else if (!m_mesh.pyramids[m_photon.curPyramid].InPyramid(m_photon.pos))
        {
            Printf_error("curPyramid: %d 设置错误, 重新计算中，此操作会耗费大量时间, 请正确预设curPyramid\n",
                         m_photon.curPyramid);
            m_photon.curPyramid = FindCurPyramid();
        }
        if (m_photon.curPyramid == -1)
        {
            //射线检测
            Printf_error(
                "光子初始位置不在mesh内部, emit需要射线检测, 此操作会耗费大量时间, 请尽量使用预设curPyramid\n");
            float min_dist         = 1e10;
            float min_dist_pyramid = -1;
            for (int i = 0; i < m_mesh.pyramids.extent(0); i++)
            {
                auto result =
                    IntersectionUtils::ray_pyramid_intersection(m_mesh.pyramids[i], m_photon.pos, m_photon.dir);
                for (int j = 0; j < 4; j++)
                {
                    if (result.result[j].hit)
                    {
                        if (result.result[j].t < min_dist)
                        {
                            min_dist         = result.result[j].t;
                            min_dist_pyramid = i;
                        }
                    }
                }
            }
            if (min_dist_pyramid == -1)
            {
                Printf_error("当前光子设置不与mesh相交, 请检查光子初始位置和方向\n");
                return false;
            }
            m_photon.curPyramid = min_dist_pyramid;
            m_photon.pos        = m_photon.pos + m_photon.dir * min_dist;
        }

        return true;
    }
    KOKKOS_INLINE_FUNCTION
    bool GetNextPyramid(Index* nextPyramid, Scalar* dist)
    {
        FUNCTION_LOG_GUARD;
        auto& curPyramid = m_photon.curPyramid;
        auto results =
            IntersectionUtils::ray_pyramid_intersection(m_mesh.pyramids[curPyramid], m_photon.pos, m_photon.dir);

        auto& result = results.result;
        for (int i = 0; i < 4; i++)
        {
            if (result[i].hit && result[i].t > 0 && result[i].t > 1e6 * REALEPS)
            {
                Point intersection_point = m_photon.pos + m_photon.dir * result[i].t;
                Point next_point_pos     = intersection_point + m_photon.dir * 0.1 * m_mesh.GetMinLength();
                *dist                    = result[i].t;

                auto& p1 = result[i].p1;
                auto& p2 = result[i].p2;
                auto& p3 = result[i].p3;
                Face hitFace(p1, p2, p3);

                if (result[i].type == 3)
                {
                    auto adjacentNum = m_mesh.adjacentPyramidsNum_3(m_photon.curPyramid);
                    KOKKOS_ASSERT(adjacentNum < m_mesh.adjacentPyramids_3.extent(1));
                    for (int j = 0; j < adjacentNum; j++)
                    {
                        auto& nextPyramid_ = m_mesh.adjacentPyramids_3(m_photon.curPyramid, j);
                        if (m_mesh.pyramids[nextPyramid_].HasFace(hitFace))
                        {
                            *nextPyramid = nextPyramid_;
                            return true;
                        }
                    }
                }
                else if (result[i].type == 2)
                {
                    auto adjacentNum = m_mesh.adjacentPyramidsNum_2(m_photon.curPyramid);
                    KOKKOS_ASSERT(adjacentNum < m_mesh.adjacentPyramids_2.extent(1));
                    auto adjacentP = m_mesh.adjacentPyramids_2(m_photon.curPyramid, adjacentNum);
                    for (int j = 0; j < adjacentNum; j++)
                    {
                        auto& nextPyramid_ = m_mesh.adjacentPyramids_2(m_photon.curPyramid, j);
                        if (m_mesh.pyramids[nextPyramid_].InPyramid(next_point_pos))
                        {
                            *nextPyramid = nextPyramid_;
                            return true;
                        }
                    }
                }
                else if (result[i].type == 1)
                {
                    auto adjacentNum = m_mesh.adjacentPyramidsNum_1(m_photon.curPyramid);
                    KOKKOS_ASSERT(adjacentNum < m_mesh.adjacentPyramids_1.extent(1));
                    auto adjacentP = m_mesh.adjacentPyramids_1(m_photon.curPyramid, adjacentNum);
                    for (int j = 0; j < adjacentNum; j++)
                    {
                        auto& nextPyramid_ = m_mesh.adjacentPyramids_1(m_photon.curPyramid, j);
                        if (m_mesh.pyramids[nextPyramid_].InPyramid(next_point_pos))
                        {
                            *nextPyramid = nextPyramid_;
                            return true;
                        }
                    }
                }
                else
                {
                    Kokkos::printf("Error: the type of the intersection is not 1, 2 or 3\n");
                }

                break;
            }  // if (results(i).hit && results(i).t > 0 && results(i).t > REALEPS)
        }
        return false;
    }
    KOKKOS_INLINE_FUNCTION
    bool MoveLen(float len)
    {
        FUNCTION_LOG_GUARD;
        m_photon.pos.x += m_photon.dir.x * len;
        m_photon.pos.y += m_photon.dir.y * len;
        m_photon.pos.z += m_photon.dir.z * len;
        return true;
    }

    KOKKOS_INLINE_FUNCTION
    bool Move()
    {
        FUNCTION_LOG_GUARD;
        Scalar s_;
        const auto& pyr                    = m_mesh.pyramids;
        const Pyramid::Attribute& cur_Attr = pyr[m_photon.curPyramid].value;
        const Scalar& mua                  = cur_Attr.mua;
        const Scalar& mus                  = cur_Attr.mus;
        const Scalar& g                    = cur_Attr.g;
        Printf("mua: %f, mus: %f, g: %f\n", mua, mus, g);
        // move the photon
        if (mua + mus > 0)
        {
            s_ = -log(GetRandom()) / (mua + mus);
        }
        else
        {
            s_ = 1;
        }
        Printf("s_: %f\n", s_);
        int max_iter = MAX_ITER;
        while (s_ >= 0 && m_photon.alive && max_iter--)
        {
            Scalar dist = 0;
            if (!GetNextPyramid(&m_photon.nextPyramid, &dist))
            {
                m_photon.alive = false;
                Kokkos::printf("[TetMesh ERROR] GetCollectType not completed\n");
                return false;
            }
            Kokkos::printf("m_photon.nextPyramid: %d\n", m_photon.nextPyramid);
            auto nowCollectType = m_collectStrategy.GetCollectType(m_photon.nextPyramid);
            switch (nowCollectType)
            {
                case CollectType::COLLECT:
                    m_photon.alive      = false;
                    result.type         = CollectType::COLLECT;
                    result.pyramidIndex = m_photon.nextPyramid;
                    result.pos          = m_photon.pos;
                    result.dir          = m_photon.dir;
                    result.weight       = m_photon.weight;
                    return true;
                case CollectType::OUTOFRANGE:
                    m_photon.alive = false;
                    result.type    = CollectType::OUTOFRANGE;
                    return true;
                    break;
                case CollectType::IGNORE: break;
                default: break;
            }
            if (s_ > dist)
            {
                Kokkos::printf("%s\n", __LINE__);
                m_photon.Ps += dist;
                MoveLen(dist);
                s_ -= dist;
                m_photon.pos = m_photon.pos + m_photon.dir * dist;
                DealWithFace();
                Kokkos::printf("%s\n", __LINE__);
            }
            else
            {
                MoveLen(s_);
                Kokkos::printf("%s\n", __LINE__);
                m_photon.Ps += s_;
                Absorb(mua, mus);
                Kokkos::printf("%s\n", __LINE__);
                Scatter(g);
                s_ = 0;
            }
            m_photon.max_z = m_photon.max_z > m_photon.pos.z ? m_photon.max_z : m_photon.pos.z;
        }
        Kokkos::printf("%s\n", __LINE__);
        return true;
    }
    KOKKOS_INLINE_FUNCTION
    bool Roulette()
    {
        FUNCTION_LOG_GUARD;
        if (m_photon.weight < 0.0001)
        {
            if (GetRandom(0, 1) > 0.1)
            {
                m_photon.alive = false;
                return false;
            }
            else
            {
                m_photon.weight /= 0.1;
                return true;
            }
        }
        else
        {
            return true;
        }
    }
    KOKKOS_INLINE_FUNCTION
    void Mirror()
    {
        FUNCTION_LOG_GUARD;
        auto n     = m_photon.nextFace.normal();
        float cdot = n.x * m_photon.dir.x + n.y * m_photon.dir.y + n.z * m_photon.dir.z;
        m_photon.dir.x -= 2.0f * cdot * n.x;
        m_photon.dir.y -= 2.0f * cdot * n.y;
        m_photon.dir.z -= 2.0f * cdot * n.z;
        m_photon.nextPyramid = m_photon.curPyramid;
    }
    KOKKOS_INLINE_FUNCTION
    void Transmit(float nipnt, float costhi, float costht, Point nor)
    {
        FUNCTION_LOG_GUARD;
        if (costhi > 0.0)
        {
            m_photon.dir.x = nipnt * m_photon.dir.x + (nipnt * costhi - costht) * nor.x;
            m_photon.dir.y = nipnt * m_photon.dir.y + (nipnt * costhi - costht) * nor.y;
            m_photon.dir.z = nipnt * m_photon.dir.z + (nipnt * costhi - costht) * nor.z;
        }
        else
        {
            m_photon.dir.x = nipnt * m_photon.dir.x + (nipnt * costhi + costht) * nor.x;
            m_photon.dir.y = nipnt * m_photon.dir.y + (nipnt * costhi + costht) * nor.y;
            m_photon.dir.z = nipnt * m_photon.dir.z + (nipnt * costhi + costht) * nor.z;
        }
        m_photon.curPyramid = m_photon.nextPyramid;
    }
    KOKKOS_INLINE_FUNCTION
    void DealWithFace()
    {
        FUNCTION_LOG_GUARD;
        auto nor    = m_photon.nextFace.normal();
        float n     = m_mesh.pyramids[m_photon.curPyramid].value.n;
        float new_n = m_mesh.pyramids[m_photon.nextPyramid].value.n;

        float nipnt = n / new_n;
        if (nipnt == 1)
        {
            m_photon.curPyramid = m_photon.nextPyramid;
            return;
        }
        float costhi = -(m_photon.dir.x * nor.x + m_photon.dir.y * nor.y + m_photon.dir.z * nor.z);
        if (1.0 - powf(nipnt, 2) * (1.0 - powf(costhi, 2)) <= 0.0)
        {
            Mirror();
            return;
        }
        float costht = sqrtf(1.0f - powf(nipnt, 2) * (1.0f - powf(costhi, 2)));
        float thi;

        if (costhi > 0.0)
            thi = acosf(costhi);
        else
            thi = acosf(-costhi);

        float tht = acosf(costht);
        float R;

        if (sinf(thi + tht) <= 1E-15)
            R = powf((nipnt - 1.0f) / (nipnt + 1.0f), 2);
        else
            R = 0.5f * (powf(sinf(thi - tht) / sinf(thi + tht), 2) + powf(tanf(thi - tht) / tanf(thi + tht), 2));

        float xi = GetRandom();

        if (xi <= R)
        {
            Mirror();
            return;
        }
        Transmit(nipnt, costhi, costht, nor);
    }
    KOKKOS_INLINE_FUNCTION
    bool Scatter(float g)
    {
        FUNCTION_LOG_GUARD;
        float xi = 0, theta = 0, phi = 0;
        auto g2 = g * g;

        // Henye-Greenstein scattering
        if (g != 0.0)
        {
            xi = GetRandom();
            if ((0.0 < xi) && (xi < 1.0))
                theta = (1.0f + g2 - powf((1.0f - g2) / (1.0f - g * (1.0f - 2.0f * xi)), 2)) / (2.0f * g);
            else
                theta = (1.0f - xi) * M_PI;
        }
        else
            theta = 2.0f * GetRandom() - 1.0f;

        phi             = 2.0f * M_PI * GetRandom();
        float& cosTheta = theta;
        // Scatter the photon
        float dxn, dyn, dzn;
        float sinTheta = sqrtf(1 - cosTheta * cosTheta);
        float sinPsi   = sinf(phi);
        float cosPsi   = cosf(phi);
        if (fabs(m_photon.dir.z) > 0.999f)
        {
            dxn = sinTheta * cosPsi;
            dyn = sinTheta * sinPsi;
            dzn = m_photon.dir.z * cosTheta / fabs(m_photon.dir.z);
        }
        else
        {
            dxn = sinTheta * (m_photon.dir.x * m_photon.dir.z * cosPsi - m_photon.dir.y * sinPsi) /
                      sqrtf(1.0f - m_photon.dir.z * m_photon.dir.z) +
                  m_photon.dir.x * cosTheta;
            dyn = sinTheta * (m_photon.dir.y * m_photon.dir.z * cosPsi + m_photon.dir.x * sinPsi) /
                      sqrtf(1.0f - m_photon.dir.z * m_photon.dir.z) +
                  m_photon.dir.y * cosTheta;
            dzn = -sinTheta * cosPsi * sqrtf(1.0f - m_photon.dir.z * m_photon.dir.z) + m_photon.dir.z * cosTheta;
        }

        float norm = sqrtf(dxn * dxn + dyn * dyn + dzn * dzn);
        dxn /= norm;
        dyn /= norm;
        dzn /= norm;

        m_photon.dir.x = dxn;
        m_photon.dir.y = dyn;
        m_photon.dir.z = dzn;
        return true;
    }
    KOKKOS_INLINE_FUNCTION
    bool Absorb(float mua, float mus)
    {
        FUNCTION_LOG_GUARD;
        float dwa = m_photon.weight * mua / (mus + mua);
        m_photon.weight -= dwa;
        return true;
    }
};

#endif
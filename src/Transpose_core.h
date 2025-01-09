#ifndef TRANSPOSE_CORE_H
#define TRANSPOSE_CORE_H
#include <iostream>
#include "Kokkos_Assert.hpp"
#include "Mesh.h"

struct Photon3D
{
    Point pos;
    Vec3f dir;
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
class DefaultEmitCollectStrategy
{
   public:
    static constexpr int EMIT = 0;
    Kokkos::UnorderedMap<Index, CollectType, ExecSpace>& collect_map;
    DefaultEmitCollectStrategy(Kokkos::UnorderedMap<Index, CollectType, ExecSpace>& collect_map) : collect_map(collect_map) {}
    
    int emitCount = 0;
    KOKKOS_FUNCTION
    CollectType GetCollectType(Index pyIndex) const
    {
        Kokkos::printf("%s\n", __LINE__);
        return collect_map.value_at(pyIndex);
    }
    KOKKOS_FUNCTION
    void emit(Photon3D* p) const
    {
        // TODO: 实现发射逻辑
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
    const DefaultEmitCollectStrategy& m_collectStrategy;
    KOKKOS_INLINE_FUNCTION
    transpose_core(const TetMesh& mesh, const DefaultEmitCollectStrategy& collectStrategy,
                   const RandPoolType& rand_pool)
        : m_mesh(mesh), m_collectStrategy(collectStrategy), m_photon(), rand_pool(rand_pool)
    {
    }
    KOKKOS_INLINE_FUNCTION
    void run()
    {
        emit();
        int i = MAX_ITER;
        checkInit();
        while (m_photon.alive && i--)
        {
            move();
            roulette();
        }

        result.type = CollectType::IGNORE;
    }
    KOKKOS_INLINE_FUNCTION
    void checkInit()
    {
        Kokkos::printf("m_photon.curPyramid: %d\n", m_photon.curPyramid);
        Kokkos::printf("m_mesh.pyramids.extent(0): %d\n", m_mesh.pyramids.extent(0));
        KOKKOS_ASSERT(m_photon.curPyramid >= -1);
        KOKKOS_ASSERT(m_photon.curPyramid < m_mesh.pyramids.extent(0));
    }
    KOKKOS_INLINE_FUNCTION
    Scalar random(Scalar lower = 0, Scalar upper = 1) { return rand_pool.get_state().drand(); }
    KOKKOS_INLINE_FUNCTION
    bool emit()
    {
        // emit a photon
        m_collectStrategy.emit(&m_photon);
        return true;
    }
    KOKKOS_INLINE_FUNCTION
    bool Get_next_Pyramid(Index* nextPyramid, Scalar* dist)
    {
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
                    auto adjacentP   = m_mesh.adjacentPyramids_2(m_photon.curPyramid, adjacentNum);
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
                    auto adjacentP   = m_mesh.adjacentPyramids_1(m_photon.curPyramid, adjacentNum);
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
    bool move_len(float len)
    {
        m_photon.pos.x += m_photon.dir.x * len;
        m_photon.pos.y += m_photon.dir.y * len;
        m_photon.pos.z += m_photon.dir.z * len;
        return true;
    }
    KOKKOS_INLINE_FUNCTION
    bool move()
    {
        Scalar s_;
        const auto& pyr                    = m_mesh.pyramids;
        auto idx                           = m_photon.curPyramid;
        const auto& pyrI                   = pyr[idx];
        const Pyramid::Attribute& cur_Attr = pyrI.value;
        const Scalar& mua                  = cur_Attr.mua;
        const Scalar& mus                  = cur_Attr.mus;
        const Scalar& g                    = cur_Attr.g;
        KOKKOS_ASSERT(idx < pyr.extent(0));
        // move the photon
        if (mua + mus > 0)
        {
            s_ = -log(random()) / (mua + mus);
        }
        else
        {
            s_ = 1;
        }

        int max_iter = MAX_ITER;
        while (s_ >= 0 && m_photon.alive && max_iter--)
        {
            Scalar dist = 0;
            if(!Get_next_Pyramid(&m_photon.nextPyramid, &dist)){
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
                move_len(dist);
                s_ -= dist;
                m_photon.pos = m_photon.pos + m_photon.dir * dist;
                DealWithFace();
                Kokkos::printf("%s\n", __LINE__);
            }
            else
            {
                move_len(s_);
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
    bool roulette()
    {
        if (m_photon.weight < 0.0001)
        {
            if (random(0, 1) > 0.1)
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

        float xi = random();

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
        float xi = 0, theta = 0, phi = 0;
        auto g2 = g * g;

        // Henye-Greenstein scattering
        if (g != 0.0)
        {
            xi = random();
            if ((0.0 < xi) && (xi < 1.0))
                theta = (1.0f + g2 - powf((1.0f - g2) / (1.0f - g * (1.0f - 2.0f * xi)), 2)) / (2.0f * g);
            else
                theta = (1.0f - xi) * M_PI;
        }
        else
            theta = 2.0f * random() - 1.0f;

        phi             = 2.0f * M_PI * random();
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
        float dwa = m_photon.weight * mua / (mus + mua);
        m_photon.weight -= dwa;
        return true;
    }
};

#endif